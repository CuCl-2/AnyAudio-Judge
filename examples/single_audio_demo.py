"""End-to-end demo for a single audio sample.

Pipeline:
  1. (text-only) decompose the instruction with Qwen3-30B-A3B-Instruct-2507
     into a list of binary rubric items, conditional on ``--subset``.
  2. (text-only) clean the rubric with the same model — drops items that
     introduce details not actually present in the caption (the
     hallucination filter from ``data_clean.py``). If every item is
     removed, falls back to a single holistic rubric whose question equals
     the original caption.
  3. (audio)     score the audio with one of AnyAudio-Judge-{7B,30B} in
     either ``logits`` mode (per-rubric soft yes-prob) or ``generate``
     mode (one-shot JSON yes/no with evidence).

All three stages use HuggingFace transformers (no vLLM dependency). The
30B decomposer is the slow part — expect tens of seconds per call on a
single GPU; this script is meant for inspection / one-off use.

Usage::

    python examples/single_audio_demo.py \\
        --audio /path/to/clip.wav \\
        --instruction "A gentle, delicate female voice ..." \\
        --subset speech \\
        --judge_model cucl2/AnyAudio-Judge-7B \\
        --mode logits
"""

from __future__ import annotations

import argparse
import json

from anyaudio_judge import AnyAudioJudge, decompose_instruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnyAudio-Judge: single-audio demo")
    parser.add_argument("--audio", required=True, help="path to a wav/mp3/flac file")
    parser.add_argument("--instruction", required=True,
                        help="caption / instruction the audio is supposed to follow")
    parser.add_argument("--subset", default="mix",
                        choices=["speech", "sound", "music", "mix"],
                        help="which decomposition prompt to use (default: mix)")
    parser.add_argument("--no_clean", action="store_true",
                        help="skip the hallucination-filter cleaning step "
                             "(by default the rubric is cleaned)")
    parser.add_argument("--decompose_model",
                        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="text LLM that produces and cleans the rubric items")
    parser.add_argument("--decompose_dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--decompose_device_map", default="auto")

    parser.add_argument("--judge_model", default="cucl2/AnyAudio-Judge-7B",
                        help="audio judge checkpoint (e.g. cucl2/AnyAudio-Judge-7B "
                             "or cucl2/AnyAudio-Judge-30B)")
    parser.add_argument("--judge_dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--judge_device_map", default="auto")
    parser.add_argument("--mode", default="logits", choices=["logits", "generate"],
                        help="logits = per-rubric soft yes-prob; "
                             "generate = one JSON call with all rubric items")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="only used by --mode generate")

    parser.add_argument("--rubric_json", default=None,
                        help="optional path to a precomputed rubric JSON; if "
                             "given, --decompose_model is skipped entirely")
    parser.add_argument("--output_json", default=None,
                        help="optional path to dump the full result as JSON")
    return parser.parse_args()


def _key(item):
    return (item.get("dimension", ""), item.get("question", ""))


def main():
    import torch

    args = parse_args()
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    # Step 1+2: rubric decomposition + (optional) cleaning
    decomposed = []
    if args.rubric_json:
        with open(args.rubric_json, "r", encoding="utf-8") as f:
            rubric = json.load(f)
        print(f">>> Loaded {len(rubric)} rubric items from {args.rubric_json}")
    else:
        print(f">>> Decomposing instruction (subset={args.subset}, clean={not args.no_clean})")
        print(f"    LM: {args.decompose_model}")
        decomposed, rubric = decompose_instruction(
            args.instruction,
            subset=args.subset,
            clean=not args.no_clean,
            model_name_or_path=args.decompose_model,
            torch_dtype=args.decompose_dtype,
            device_map=args.decompose_device_map,
            return_intermediate=True,
        )
        print(f"    decomposed: {len(decomposed)} items")
        if not args.no_clean:
            removed = [d for d in decomposed if _key(d) not in {_key(r) for r in rubric}]
            print(f"    cleaned   : {len(rubric)} items  (removed {len(removed)})")
            if not decomposed and not removed:
                pass  # nothing to report
            elif rubric and rubric[0].get("dimension") == "整体" and rubric[0].get("question") == args.instruction:
                print("    (cleaning removed every item; using holistic fallback rubric)")
            for d in removed:
                print(f"      ✗ removed: ({d.get('dimension','')}) {d.get('question','')}")

    if not rubric:
        raise SystemExit("rubric is empty after decomposition; aborting.")

    print()
    for i, item in enumerate(rubric):
        print(f"    [{i}] ({item.get('dimension','')}) {item.get('question','')}")
        if item.get("basis"):
            print(f"          ↳ basis: {item['basis']}")

    # Step 3: judge
    print(f"\n>>> Loading judge: {args.judge_model}")
    judge = AnyAudioJudge.from_pretrained(
        args.judge_model,
        torch_dtype=dtype_map[args.judge_dtype],
        device_map=args.judge_device_map,
    )

    print(f">>> Judging audio: {args.audio} (mode={args.mode})")
    result = judge.judge(
        audio_path=args.audio,
        rubric=rubric,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\nAlignment score = {result.score:.4f}  (mode={result.mode})")
    print(f"{'#':>3}  {'answer':<8}  {'p_yes':>7}  question")
    print("-" * 80)
    for it in result.items:
        marker = "✓" if it.answer == "yes" else ("✗" if it.answer == "no" else "?")
        p = "  nan" if it.p_yes != it.p_yes else f"{it.p_yes:6.3f}"
        print(f"{it.id:>3}  {marker} {it.answer:<6}  {p}  ({it.dimension}) {it.question}")
        if it.evidence:
            print(f"     evidence: {it.evidence}")

    if args.output_json:
        payload = {
            "audio": args.audio,
            "instruction": args.instruction,
            "subset": args.subset,
            "judge_model": args.judge_model,
            "mode": args.mode,
            "score": result.score,
            "rubric_decomposed": decomposed,
            "rubric_used": rubric,
            "items": [it.to_dict() for it in result.items],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved detailed result to {args.output_json}")


if __name__ == "__main__":
    main()
