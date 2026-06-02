"""Run AnyAudio-Judge across the full AnyAudio-Judge Bench.

The benchmark dataset already ships pre-decomposed rubric items in the
``rubric`` column, so we skip the (expensive) text-LLM decomposition step
entirely and feed those rubric items straight to the audio judge.

Two scoring modes mirror those in single-audio inference:

* ``--mode logits`` (default) — per-rubric soft yes-prob; sample score is
  the mean of ``p_yes`` across rubric items. We **do not** binarize this
  score by default; it is intended as a continuous reward signal.

* ``--mode generate`` — one JSON call per audio; sample score is the yes
  ratio. We binarize it against ``--yes_threshold`` (default 0.8) to
  compute accuracy against the gold yes/no label.

Usage::

    python evaluate_benchmark.py \\
        --judge_model cucl2/AnyAudio-Judge-7B \\
        --languages en zh \\
        --mode logits \\
        --output_dir ./eval_results
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm.auto import tqdm

from anyaudio_judge import AnyAudioJudge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnyAudio-Judge benchmark evaluation")
    parser.add_argument("--judge_model", default="cucl2/AnyAudio-Judge-7B",
                        help="HF id of the judge checkpoint")
    parser.add_argument("--judge_dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--judge_device_map", default="auto")

    parser.add_argument("--benchmark_repo", default="cucl2/AnyAudio-Judge-Bench",
                        help="HF dataset id of the benchmark")
    parser.add_argument("--languages", nargs="+", default=["en", "zh"],
                        choices=["en", "zh"])
    parser.add_argument("--split", default="test")

    parser.add_argument("--mode", default="logits", choices=["logits", "generate"])
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="(generate mode) max tokens for the JSON output")
    parser.add_argument("--yes_threshold", type=float, default=0.8,
                        help="(generate mode) yes_ratio >= threshold → predict yes "
                             "(default 0.8). Adjust as needed.")

    parser.add_argument("--limit", type=int, default=None,
                        help="cap on samples per language (sanity runs)")
    parser.add_argument("--output_dir", default="./eval_outputs")
    return parser.parse_args()


def _materialize_audio(audio_field) -> str:
    """The benchmark stores audio as embedded bytes. We need a real path
    for the judge's audio loader, so spill to a tmp file when necessary."""
    if isinstance(audio_field, dict):
        path = audio_field.get("path")
        if path and os.path.exists(path):
            return path
        arr = audio_field.get("array")
        sr = audio_field.get("sampling_rate")
        if arr is not None and sr:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, arr, sr)
            return tmp.name
    return str(audio_field)


def _normalize_rubric(rubric):
    """Datasets serialises the rubric column as a list of dicts; some rows
    have empty rubric (no decomposition was found)."""
    if rubric is None:
        return []
    out = []
    for it in rubric:
        if not isinstance(it, dict):
            continue
        out.append({
            "dimension": it.get("dimension") or it.get("维度") or "",
            "question": it.get("question") or it.get("题目") or "",
            "basis": it.get("basis") or it.get("依据") or "",
        })
    return out


def main():
    import torch

    args = parse_args()
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> Loading judge: {args.judge_model}")
    judge = AnyAudioJudge.from_pretrained(
        args.judge_model,
        torch_dtype=dtype_map[args.judge_dtype],
        device_map=args.judge_device_map,
    )

    summary: dict = {
        "judge_model": args.judge_model,
        "mode": args.mode,
        "yes_threshold": args.yes_threshold if args.mode == "generate" else None,
        "languages": {},
    }

    for lang in args.languages:
        print(f"\n=== Language: {lang} ===")
        ds = load_dataset(args.benchmark_repo, lang, split=args.split)
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        print(f"    {len(ds)} samples")

        per_subset_scores: dict = defaultdict(list)
        per_subset_correct: dict = defaultdict(list)  # only for generate mode
        records_path = out_dir / f"records_{lang}.jsonl"

        with open(records_path, "w", encoding="utf-8") as out_f:
            for sample in tqdm(ds, desc=f"[{lang}] judging"):
                rubric = _normalize_rubric(sample.get("rubric"))
                if not rubric:
                    # nothing to score; still write a row so users know
                    record = {
                        "uuid": sample["uuid"],
                        "subset": sample["subset"],
                        "label": sample["label"],
                        "score": float("nan"),
                        "n_rubric": 0,
                        "skipped": "empty rubric",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                audio_path = _materialize_audio(sample["audio"])
                try:
                    result = judge.judge(
                        audio_path=audio_path,
                        rubric=rubric,
                        mode=args.mode,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as exc:
                    record = {
                        "uuid": sample["uuid"],
                        "subset": sample["subset"],
                        "label": sample["label"],
                        "score": float("nan"),
                        "n_rubric": len(rubric),
                        "error": str(exc),
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                record = {
                    "uuid": sample["uuid"],
                    "subset": sample["subset"],
                    "label": sample["label"],
                    "score": result.score,
                    "n_rubric": len(rubric),
                    "items": [it.to_dict() for it in result.items],
                }
                # binarized prediction is only computed for generate mode
                if args.mode == "generate":
                    if result.score == result.score:  # not nan
                        record["pred"] = "yes" if result.score >= args.yes_threshold else "no"
                        record["correct"] = int(record["pred"] == sample["label"])
                        per_subset_correct[sample["subset"]].append(record["correct"])
                if result.score == result.score:
                    per_subset_scores[sample["subset"]].append(result.score)

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Build per-language report
        lang_report = {"per_subset_mean_score": {}, "n_samples": len(ds)}
        for subset in sorted(per_subset_scores):
            lang_report["per_subset_mean_score"][subset] = float(np.mean(per_subset_scores[subset]))
        lang_report["overall_mean_score"] = float(np.mean(
            [s for vals in per_subset_scores.values() for s in vals]
        )) if per_subset_scores else float("nan")

        if args.mode == "generate":
            lang_report["per_subset_acc"] = {
                subset: float(100 * np.mean(corrects))
                for subset, corrects in sorted(per_subset_correct.items())
            }
            all_corrects = [c for vals in per_subset_correct.values() for c in vals]
            lang_report["overall_acc"] = float(100 * np.mean(all_corrects)) if all_corrects else float("nan")

        summary["languages"][lang] = lang_report
        print(f"    overall mean score = {lang_report['overall_mean_score']:.4f}")
        if args.mode == "generate":
            print(f"    overall ACC (threshold={args.yes_threshold}) = {lang_report['overall_acc']:.2f}")
        for subset, sc in lang_report["per_subset_mean_score"].items():
            extra = ""
            if args.mode == "generate":
                acc = lang_report["per_subset_acc"].get(subset)
                if acc is not None:
                    extra = f"  ACC={acc:5.2f}"
            print(f"      {subset:<12} mean={sc:.4f}{extra}  n={len(per_subset_scores[subset])}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
