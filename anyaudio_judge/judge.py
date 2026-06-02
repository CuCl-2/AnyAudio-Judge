"""Audio judge for AnyAudio-Judge.

Two scoring modes are supported:

* ``mode='logits'`` — runs one forward pass per rubric item, asks the model
  for a single yes/no token, and reads the logits at the answer position.
  ``p_yes = softmax(logit_yes, logit_no)`` gives a soft, fine-grained score.
  ``score = mean(p_yes)`` over the rubric. This is the default and matches
  the paper's headline numbers.

* ``mode='generate'`` — packs all rubric items into one prompt, lets the
  model emit a JSON array ``[{id, answer, evidence}, …]`` in a single
  generation, and counts the yes ratio. Faster (1 forward pass) at the
  cost of losing soft-probability information.

Audio pre-processing (mono, 16 kHz resampling) is hidden inside the class:
callers only pass file paths.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from .prompts import (
    JUDGE_GENERATE_SYSTEM_PROMPT,
    JUDGE_LOGITS_SYSTEM_PROMPT,
    build_generate_user_text,
    build_logits_user_text,
)


_TARGET_SR = 16000


# ── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class RubricResult:
    id: int
    dimension: str
    question: str
    answer: str          # 'yes' / 'no' / 'unparsed'
    p_yes: float         # in [0, 1]; for generate mode, 1.0/0.0/None depending on answer
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class JudgeResult:
    score: float
    mode: str
    items: List[RubricResult]

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "mode": self.mode,
                "items": [it.to_dict() for it in self.items]}


# ── Audio helpers ────────────────────────────────────────────────────────────

def _load_audio(audio_path: str, target_sr: int = _TARGET_SR) -> np.ndarray:
    """Load any audio file, force mono + ``target_sr``, return float32 array."""
    import soundfile as sf

    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        n = int(len(audio) * target_sr / sr)
        idx = np.linspace(0, len(audio) - 1, n)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


# ── JSON parser for generate mode (ported from eval_once.py) ────────────────

def _parse_generate_answers(raw: str, n: int):
    """Return list of (answer, evidence) tuples of length n."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

    out = [(None, "")] * n
    try:
        arr = json.loads(cleaned)
        if isinstance(arr, list):
            for item in arr:
                if not isinstance(item, dict):
                    continue
                idx = item.get("id")
                if not isinstance(idx, int) or idx < 0 or idx >= n:
                    continue
                ans = str(item.get("answer", "")).strip().lower()
                if ans not in ("yes", "no"):
                    ans = None
                out[idx] = (ans, str(item.get("evidence", "")))
            return out
    except json.JSONDecodeError:
        pass

    # fallback: scan answers/evidence in order
    answers = re.findall(r'"answer"\s*:\s*"(yes|no)"', raw, flags=re.IGNORECASE)
    evidences = re.findall(r'"evidence"\s*:\s*"([^"]*)"', raw)
    for i in range(n):
        a = answers[i].lower() if i < len(answers) else None
        e = evidences[i] if i < len(evidences) else ""
        out[i] = (a, e)
    return out


# ── The judge ────────────────────────────────────────────────────────────────

class AnyAudioJudge:
    """Audio judge that scores (audio, rubric) pairs.

    Loads a Qwen2.5-Omni / Qwen3-Omni-style model via ``transformers`` and
    keeps it cached for repeated calls.
    """

    def __init__(self, model, processor, *, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        self.model = model
        self.processor = processor
        self.torch_dtype = torch_dtype
        self.device = next(self.model.parameters()).device
        self._yes_token_ids = self._encode_yesno_tokens("yes")
        self._no_token_ids = self._encode_yesno_tokens("no")

    # ---- constructor ----

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Union[str, Dict[str, Any]] = "auto",
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "AnyAudioJudge":
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        model.eval()
        return cls(model, processor, torch_dtype=torch_dtype)

    # ---- helpers ----

    def _tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    def _encode_yesno_tokens(self, word: str) -> List[int]:
        """Collect every token id that decodes to ``word`` after stripping
        surrounding whitespace. We try several variants so the lookup also
        finds e.g. " yes" / "Yes"."""
        tok = self._tokenizer()
        ids = set()
        for variant in (word, " " + word, word.capitalize(), " " + word.capitalize(), word.upper(), " " + word.upper()):
            enc = tok.encode(variant, add_special_tokens=False)
            for tid in enc:
                # keep ids that decode to the same single word ignoring whitespace
                decoded = tok.decode([tid]).strip().lower()
                if decoded == word.lower():
                    ids.add(int(tid))
        if not ids:
            # fallback: take whatever the tokenizer assigns to a bare " yes" / " no"
            ids.update(int(t) for t in tok.encode(" " + word, add_special_tokens=False))
        return sorted(ids)

    def _build_inputs(self, system: str, user_text: str, audio: np.ndarray):
        """Build model inputs for an audio + system + user-text turn.

        We follow Qwen-Omni's conversational format. The processor handles
        the audio encoding; everything is sent to the model device.
        """
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "<audio>"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        # Render to a templated text string (multimodal placeholders are
        # handled by the processor when we pass `audios=`).
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        try:
            inputs = self.processor(
                text=[text],
                audio=[audio],
                sampling_rate=_TARGET_SR,
                return_tensors="pt",
                padding=True,
            )
        except TypeError:
            # Older Qwen-Omni processors used `audios=` instead of `audio=`.
            inputs = self.processor(
                text=[text],
                audios=[audio],
                sampling_rate=_TARGET_SR,
                return_tensors="pt",
                padding=True,
            )
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # ---- mode A: per-rubric logits ----

    @torch.no_grad()
    def _score_logits(self, audio: np.ndarray, rubric: Sequence[Dict[str, str]]) -> List[RubricResult]:
        items: List[RubricResult] = []
        for idx, item in enumerate(rubric):
            user_text = build_logits_user_text(item)
            inputs = self._build_inputs(JUDGE_LOGITS_SYSTEM_PROMPT, user_text, audio)

            # Single-step generation; we only care about the next-token logits.
            out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
            scores = out.scores[0][0].float()  # [vocab]

            # Aggregate logits across all token-id variants of yes/no.
            yes_logit = scores[self._yes_token_ids].max().item() if self._yes_token_ids else -1e9
            no_logit = scores[self._no_token_ids].max().item() if self._no_token_ids else -1e9

            # Two-way softmax for stability.
            m = max(yes_logit, no_logit)
            ey = np.exp(yes_logit - m)
            en = np.exp(no_logit - m)
            p_yes = float(ey / (ey + en))

            answer = "yes" if p_yes >= 0.5 else "no"
            items.append(RubricResult(
                id=idx,
                dimension=item.get("dimension", item.get("维度", "")),
                question=item.get("question", item.get("题目", "")),
                answer=answer,
                p_yes=p_yes,
                evidence="",
            ))
        return items

    # ---- mode B: one-shot generate ----

    @torch.no_grad()
    def _score_generate(
        self,
        audio: np.ndarray,
        rubric: Sequence[Dict[str, str]],
        max_new_tokens: int = 4096,
    ) -> List[RubricResult]:
        user_text = build_generate_user_text(rubric)
        inputs = self._build_inputs(JUDGE_GENERATE_SYSTEM_PROMPT, user_text, audio)

        prompt_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        seqs = out.sequences if hasattr(out, "sequences") else out
        text = self._tokenizer().decode(seqs[0, prompt_len:], skip_special_tokens=True)
        parsed = _parse_generate_answers(text, len(rubric))

        items: List[RubricResult] = []
        for idx, (item, (ans, ev)) in enumerate(zip(rubric, parsed)):
            if ans == "yes":
                p_yes = 1.0
                answer = "yes"
            elif ans == "no":
                p_yes = 0.0
                answer = "no"
            else:
                p_yes = float("nan")
                answer = "unparsed"
            items.append(RubricResult(
                id=idx,
                dimension=item.get("dimension", item.get("维度", "")),
                question=item.get("question", item.get("题目", "")),
                answer=answer,
                p_yes=p_yes,
                evidence=ev or "",
            ))
        return items

    # ---- public API ----

    def judge(
        self,
        audio_path: Union[str, np.ndarray],
        rubric: Sequence[Dict[str, str]],
        *,
        mode: str = "logits",
        max_new_tokens: int = 4096,
    ) -> JudgeResult:
        if mode not in {"logits", "generate"}:
            raise ValueError(f"mode must be 'logits' or 'generate', got {mode!r}")
        if not rubric:
            return JudgeResult(score=float("nan"), mode=mode, items=[])

        audio = audio_path if isinstance(audio_path, np.ndarray) else _load_audio(audio_path)

        if mode == "logits":
            items = self._score_logits(audio, rubric)
            valid = [it.p_yes for it in items if not np.isnan(it.p_yes)]
            score = float(np.mean(valid)) if valid else float("nan")
        else:
            items = self._score_generate(audio, rubric, max_new_tokens=max_new_tokens)
            yes_no = [it for it in items if it.answer in ("yes", "no")]
            if yes_no:
                yes_count = sum(1 for it in yes_no if it.answer == "yes")
                score = float(yes_count / len(yes_no))
            else:
                score = float("nan")

        return JudgeResult(score=score, mode=mode, items=items)
