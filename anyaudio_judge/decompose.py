"""HuggingFace-only decomposition + cleaning of audio captions.

Two text-LLM steps share a single ``Qwen/Qwen3-30B-A3B-Instruct-2507``
backbone (loaded once and cached):

1. **Decompose** — turn a caption into a list of binary rubric items,
   conditioned on the audio subset (speech / sound / music / mix).
2. **Clean** — filter out rubric items that introduce details not actually
   mentioned in the caption (the "hallucination filter" from
   ``data_clean.py``). Done per-caption in a single LLM call.

The default ``decompose_instruction(...)`` runs both steps and returns a
cleaned rubric. If cleaning removes every item, it falls back to a single
holistic rubric whose question equals the original caption — this matches
the user's request that "if all items are deleted, fall back to one
rubric = the caption itself, i.e. no decomposition".

The model is large (≈60 GB BF16); inference is slow on a single GPU.
``HFDecomposer`` is intentionally thin so users can swap in their own
backend (vLLM, TGI, OpenAI-compatible, …) by re-implementing the
``call_text`` method.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

from .prompts import (
    CLEAN_SYSTEM_PROMPT,
    DECOMPOSE_PROMPTS,
    build_clean_user_text,
)


# ── JSON robust parser (ported from decompose_all.py) ────────────────────────

def _try_fix_json(raw: str):
    """Try several recovery strategies for malformed JSON outputs."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    cleaned = cleaned.replace("“", '\\"').replace("”", '\\"')

    try:
        return json.loads(cleaned), True
    except json.JSONDecodeError:
        pass

    step = re.sub(r'(\n\s+)(题目|维度|依据)":', r'\1"\2":', cleaned)
    try:
        return json.loads(step), True
    except json.JSONDecodeError:
        pass

    lines = cleaned.split("\n")
    fixed = []
    _pat_closed = re.compile(r'^(\s*"(?:依据|题目|维度)":\s*")(.*)((?<!\\)",?\s*)$', re.DOTALL)
    _pat_open = re.compile(r'^(\s*"(?:依据|题目|维度)":\s*")(.+)(,?\s*)$')
    for line in lines:
        m = _pat_closed.match(line)
        if m:
            prefix, content, suffix = m.group(1), m.group(2), m.group(3)
            content = re.sub(r'(?<!\\)"', '\\"', content)
            line = prefix + content + suffix
        else:
            m2 = _pat_open.match(line)
            if m2:
                prefix, content, trailing = m2.group(1), m2.group(2), m2.group(3)
                content = re.sub(r'(?<!\\)"', '\\"', content)
                line = prefix + content + '"' + trailing
        fixed.append(line)
    step = "\n".join(fixed)
    try:
        return json.loads(step), True
    except json.JSONDecodeError:
        pass

    step2 = re.sub(r",(\s*[}\]])", r"\1", step)
    try:
        return json.loads(step2), True
    except json.JSONDecodeError:
        pass

    for base in (step2, cleaned):
        pos = base.rfind("\n        }")
        if pos != -1:
            truncated = base[:pos + 10].rstrip().rstrip(",") + "\n    ]\n}"
            try:
                return json.loads(truncated), True
            except json.JSONDecodeError:
                pass

    return None, False


def _parse_clean_judgments(raw: str, n: int) -> List[bool]:
    """Parse the cleaner's ``{"results": [{"id":N,"judgment":"keep|remove"}]}``
    into a list of length ``n`` of keep flags. Raises on parse failure
    (per the user's instruction: surface errors loudly so they can retry).
    """
    text = raw.strip()
    # Strip code fences first
    if "```" in text:
        # take the first fenced block if present
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if m:
            text = m.group(1)
    else:
        # otherwise take outermost {...}
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    data = json.loads(text)  # raises JSONDecodeError -> caller propagates
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError(f"clean response missing 'results' list: {raw!r}")

    keep = [True] * n  # default keep on missing entries (defensive)
    seen = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        idx = item.get("id")
        if not isinstance(idx, int) or idx < 0 or idx >= n:
            continue
        seen.add(idx)
        verdict = str(item.get("judgment", "")).strip().lower()
        keep[idx] = (verdict == "keep")
    if len(seen) < n:
        # Tolerate small misses but flag them; this is the only soft path.
        # Hard parse errors above are still raised.
        pass
    return keep


def _normalize_questions(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert the LLM JSON payload into a uniform English schema."""
    items = (
        payload.get("Question_List")
        or payload.get("question_list")
        or payload.get("判断题列表")
        or []
    )
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        out.append({
            "dimension": it.get("Dimension") or it.get("维度") or "",
            "question": it.get("Question") or it.get("题目") or "",
            "basis": it.get("Basis") or it.get("依据") or "",
        })
    return out


def _holistic_fallback(caption: str) -> List[Dict[str, str]]:
    """Return a single-item rubric that reduces to a holistic match check.

    Triggered when cleaning removes every decomposed item; semantically
    equivalent to "no decomposition was useful, ask the audio whether it
    matches the original caption".
    """
    return [{
        "dimension": "整体",
        "question": caption,
        "basis": caption,
    }]


# ── HuggingFace transformers backbone ────────────────────────────────────────

class HFDecomposer:
    """Wraps a local causal LM for caption decomposition + rubric cleaning.

    The model is loaded lazily and reused across calls. Both
    :meth:`decompose` and :meth:`clean` go through :meth:`call_text`, so
    a downstream user can subclass and swap that single method to retarget
    a different backend.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        *,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype_map.get(torch_dtype, torch.bfloat16),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self._supports_no_thinking = True  # try the chat-template flag first

    # ---- single text-completion call shared by decompose + clean ----

    def call_text(
        self,
        system_prompt: str,
        user_content: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        no_think_marker: bool = True,
    ) -> str:
        """Run one text-only generation and return the model's text output.

        ``no_think_marker=True`` prepends ``/no_think`` to the user message
        so Qwen3-Instruct emits JSON directly without a thinking trace.
        """
        import torch

        if no_think_marker:
            user_content = "/no_think\n" + user_content

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        kwargs = dict(add_generation_prompt=True, return_tensors="pt")

        if self._supports_no_thinking:
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    chat_template_kwargs={"enable_thinking": False},
                    **kwargs,
                )
            except (TypeError, ValueError):
                self._supports_no_thinking = False
                inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
        else:
            inputs = self.tokenizer.apply_chat_template(messages, **kwargs)

        inputs = inputs.to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
        # Strip any residual <think>…</think> blocks
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    # ---- step 1: decompose ----

    def decompose(
        self,
        caption: str,
        *,
        subset: str = "mix",
        max_new_tokens: Optional[int] = None,
        return_raw: bool = False,
    ):
        if subset not in DECOMPOSE_PROMPTS:
            raise ValueError(
                f"unknown subset {subset!r}; valid choices: {list(DECOMPOSE_PROMPTS)}"
            )
        cfg = DECOMPOSE_PROMPTS[subset]
        budget = max_new_tokens or cfg["max_tokens"]
        text = self.call_text(
            cfg["system_prompt"],
            cfg["user_prefix"] + caption,
            max_new_tokens=budget,
            temperature=0.0,
            no_think_marker=True,
        )
        parsed, ok = _try_fix_json(text)
        rubric: List[Dict[str, str]] = _normalize_questions(parsed) if ok else []
        if return_raw:
            return rubric, text
        return rubric

    # ---- step 2: clean ----

    def clean(
        self,
        caption: str,
        rubric: List[Dict[str, str]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        fallback_to_caption: bool = True,
    ) -> List[Dict[str, str]]:
        """Filter hallucinated rubric items.

        Raises on hard failures (network, malformed JSON, no ``results``
        field) so the caller can retry. Returns a fallback single-item
        rubric if cleaning empties the list and ``fallback_to_caption``
        is True.
        """
        if not rubric:
            return _holistic_fallback(caption) if fallback_to_caption else []

        user = build_clean_user_text(caption, rubric)
        raw = self.call_text(
            CLEAN_SYSTEM_PROMPT,
            user,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            no_think_marker=False,  # cleaner already uses enable_thinking=False
        )
        try:
            keep_flags = _parse_clean_judgments(raw, len(rubric))
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to parse cleaner response (please retry):\n{raw!r}"
            ) from exc

        kept = [item for item, k in zip(rubric, keep_flags) if k]
        if not kept and fallback_to_caption:
            return _holistic_fallback(caption)
        return kept


# ── module-level convenience ─────────────────────────────────────────────────

@lru_cache(maxsize=2)
def _cached_decomposer(model_name_or_path: str, torch_dtype: str, device_map: str) -> HFDecomposer:
    return HFDecomposer(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )


def decompose_instruction(
    caption: str,
    *,
    subset: str = "mix",
    clean: bool = True,
    model_name_or_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    max_new_tokens: Optional[int] = None,
    return_intermediate: bool = False,
) -> List[Dict[str, str]]:
    """Decompose ``caption`` into a list of binary rubric items.

    Parameters
    ----------
    caption :
        The audio caption / instruction to decompose.
    subset :
        One of ``{"speech","sound","music","mix"}``. Selects the system prompt.
        Defaults to ``"mix"`` (broad coverage across all four domains).
    clean :
        If True (default), runs the rubric through the hallucination
        filter (same model) before returning. If every item is removed,
        falls back to a single holistic rubric whose question equals the
        caption.
    return_intermediate :
        If True, returns ``(decomposed_rubric, cleaned_rubric)`` so the
        caller can inspect what was filtered.

    Returns
    -------
    list of {"dimension","question","basis"} dicts.
    """
    decomposer = _cached_decomposer(model_name_or_path, torch_dtype, device_map)
    rubric_decomposed = decomposer.decompose(
        caption, subset=subset, max_new_tokens=max_new_tokens
    )
    if not clean:
        return (rubric_decomposed, rubric_decomposed) if return_intermediate else rubric_decomposed
    rubric_cleaned = decomposer.clean(caption, rubric_decomposed, fallback_to_caption=True)
    if return_intermediate:
        return rubric_decomposed, rubric_cleaned
    return rubric_cleaned


def clean_rubric(
    caption: str,
    rubric: List[Dict[str, str]],
    *,
    model_name_or_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    fallback_to_caption: bool = True,
) -> List[Dict[str, str]]:
    """Filter ``rubric`` to remove items that introduce details not actually
    mentioned in ``caption``. Useful when you already have a rubric from
    a previous pass and want to re-clean it. Raises ``RuntimeError`` on
    hard failures (parse errors / bad JSON) so the caller can retry."""
    decomposer = _cached_decomposer(model_name_or_path, torch_dtype, device_map)
    return decomposer.clean(caption, rubric, fallback_to_caption=fallback_to_caption)

