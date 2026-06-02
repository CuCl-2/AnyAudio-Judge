# AnyAudio-Judge

> **A Dynamic Rubric-Based Benchmark and Evaluator for Audio Instruction Following**

AnyAudio-Judge introduces a *dynamic rubric-based evaluation paradigm* for instruction-guided audio generation. Instead of asking a judge for a single holistic match/mismatch decision, AnyAudio-Judge dynamically decomposes each instruction into a variable number of independent, verifiable binary rubric items, evaluates each item, and aggregates the item-level probabilities into an interpretable alignment score.

This repository accompanies our paper and releases:

- **AnyAudio-Judge Bench** — a bilingual (English / Chinese), multi-domain benchmark of **7,920** samples per language across **7** subsets (`speech`, `speech_gen`, `sound`, `sound_gen`, `music`, `music_gen`, `mix`), with deliberately constructed hard negatives.
- **AnyAudio-Judge Corpus** — a **105K**-sample SFT training set with explicit Chain-of-Thought (CoT) rationales and per-rubric binary labels.
- **AnyAudio-Judge Models** — two trained evaluators:
  - `AnyAudio-Judge-7B`  (initialized from Qwen2.5-Omni-7B)
  - `AnyAudio-Judge-30B` (initialized from Qwen3-Omni-30B-A3B-Captioner; the model reported in the paper)
- **Inference / Evaluation Code** — scripts that reproduce rubric decomposition, judging, and benchmark scoring.

---

## Resources

| Resource | HuggingFace |
|---|---|
| Benchmark (en + zh) | [`cucl2/AnyAudio-Judge-Bench`](https://huggingface.co/datasets/cucl2/AnyAudio-Judge-Bench) |
| Corpus              | [`cucl2/AnyAudio-Judge-Corpus`](https://huggingface.co/datasets/cucl2/AnyAudio-Judge-Corpus) |
| Model (7B)          | [`cucl2/AnyAudio-Judge-7B`](https://huggingface.co/cucl2/AnyAudio-Judge-7B) |
| Model (30B)         | [`cucl2/AnyAudio-Judge-30B`](https://huggingface.co/cucl2/AnyAudio-Judge-30B) |

---

## What does the dynamic rubric paradigm look like?

Given an audio-instruction pair `(a, i)`:

1. **Decompose** instruction `i` into `n` atomic, verifiable yes/no rubric items `{p_1, ..., p_n}` via an LLM (Qwen3-30B-A3B-Instruct-2507 in the paper).
2. **Judge** each `p_j` with an audio-language model. We read the logits of the answer tokens "yes" and "no" and obtain a soft satisfaction probability:
   ```
   p_j^yes = exp(z_yes) / (exp(z_yes) + exp(z_no))
   ```
3. **Aggregate** to an overall alignment score `s = (1/n) * sum_j p_j^yes`.

Each rubric item also comes with a short evidence string, giving the user an interpretable, item-level diagnosis of *which* aspect of the instruction the audio satisfies or fails.

---

## Highlights of AnyAudio-Judge Bench

| Subset       | # Samples (per language) | Description |
|--------------|--------------------------|-------------|
| `speech`     | 1,200 | Real speech, filtered from InstructTTSEval |
| `speech_gen` | 2,000 | Synthesized by Qwen3-TTS / MOSS-VoiceGen / MiMo-Audio |
| `sound`      | 1,000 | Real sound effects from Clotho v2 |
| `sound_gen`  | 1,200 | Synthesized by AudioGen / AudioLDM2 / Stable Audio |
| `music`      |   720 | Real music from Song Describer |
| `music_gen`  |   800 | Synthesized by MusicGen / ACE-Step / Stable Audio |
| `mix`        | 1,000 | Real cinematic mixed audio (~1 min clips) |
| **Total**    | **7,920** ×2 (en/zh) | |

For every subset, the positive : negative ratio is strictly **1 : 1**. Negatives are constructed by:
- **Instruction Swapping** — interchanging instructions across samples to create clear semantic mismatches.
- **Attribute Perturbation** — using an LLM to alter specific details (dialect, emotion, instrument, secondary sounds, etc.) to simulate fine-grained generation failures.

Each row in the released benchmark contains:
```
uuid       : str   # unique sample id
audio      : Audio # huggingface Audio() field, audio bytes embedded
caption    : str   # the (possibly perturbed/swapped) instruction
label      : str   # "yes" if audio matches caption, "no" otherwise
type       : str   # one of {pos, pos_clap, pos_gemini, neg_swap, neg_change, neg_gemini}
subset     : str   # one of {speech, speech_gen, sound, sound_gen, music, music_gen, mix}
rubric     : list  # list of {dimension, question, basis} (decomposed binary items, in Chinese)
```

---

## Headline Results

### AnyAudio-Judge Bench (accuracy ↑)

| Model                              | Prompt          | English Avg | Chinese Avg |
|------------------------------------|-----------------|------------:|------------:|
| Audio-Flamingo3                    | dynamic rubric  | 64.19 | 63.91 |
| MiDashengLM                        | dynamic rubric  | 67.94 | 68.35 |
| Kimi-Audio-7B-Instruct             | dynamic rubric  | 70.81 | 70.84 |
| Qwen2.5-Omni-7B                    | dynamic rubric  | 72.24 | 71.93 |
| Qwen3-Omni-30B-A3B-Instruct        | dynamic rubric  | 77.34 | 76.82 |
| Qwen3-Omni-30B-A3B-Captioner       | dynamic rubric  | 76.77 | 76.66 |
| Gemini-2.5-Pro                     | holistic        | 77.72 | 80.01 |
| **AnyAudio-Judge (this work)**     | **dynamic rubric** | **84.45** | **85.26** |

---

## Installation

```bash
# Python ≥ 3.10 is recommended.
git clone https://github.com/CuCl-2/AnyAudio-Judge.git
cd AnyAudio-Judge
pip install -r requirements.txt
```

`requirements.txt` covers the minimum dependencies for inference:
- `torch >= 2.4`
- `transformers >= 4.46` (for Qwen-Omni support)
- `accelerate`
- `librosa`, `soundfile`
- `datasets`, `huggingface_hub`
- `tqdm`, `numpy`, `pandas`

For decomposition we use Qwen3-30B-A3B-Instruct-2507 via vLLM or any OpenAI-compatible endpoint; see `decompose.py`.

---

## Quick Start

The pipeline is a three-stage process; the first two stages share one LM:

```
caption  ──[1. decompose, subset prompt ]──► raw rubric
            (Qwen3-30B-A3B-Instruct-2507)        │
                                                 ▼
                                  [2. clean, hallucination filter]
                                                 │  (same LM)
                                                 ▼
                                              cleaned rubric
audio    ──[3. judge, AnyAudio-Judge-{7B,30B}]──► alignment score
            (mode = logits | generate)
```

All stages run on **HuggingFace transformers** (no vLLM dependency).

### 1. Decompose + clean an instruction into rubric items

```python
from anyaudio_judge import decompose_instruction

caption = "A gentle, delicate female voice, with soft and smooth pitch, calm and restrained throughout."
rubric = decompose_instruction(
    caption,
    subset="speech",                                 # speech | sound | music | mix (default: mix)
    clean=True,                                      # default: True (filter hallucinated items)
    model_name_or_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
)
for item in rubric:
    print(item)
# → {"dimension": "性别", "question": "说话人是否为女性？", "basis": "female voice"}
# → ...
```

The four `subset` choices select different decomposition system prompts (speech-focused, sound-effect, music, or fully mixed audio). Pick the one that best matches the audio domain; `mix` is a safe default that works across all four.

By default, the rubric is then re-checked by the same LM and any item that introduces details not actually present in the caption is removed (the "hallucination filter" from the original `data_clean.py`). If every item is removed, `decompose_instruction` falls back to a single holistic rubric whose question is the original caption — equivalent to "no useful decomposition; ask the audio whether it matches the caption". Pass `clean=False` to skip this step or use the standalone `clean_rubric(caption, rubric)` function on a pre-existing rubric.

### 2. Judge an audio with the rubric

Two scoring modes are supported:

```python
from anyaudio_judge import AnyAudioJudge

judge = AnyAudioJudge.from_pretrained("cucl2/AnyAudio-Judge-7B")  # or -30B

# Mode A — per-rubric soft yes-prob (default; matches the paper).
result = judge.judge("./demo.wav", rubric, mode="logits")
print("alignment_score:", result.score)
for it in result.items:
    print(f"  [{it.id}] p_yes={it.p_yes:.3f}  ({it.dimension}) {it.question}")

# Mode B — single-call JSON yes/no with evidence.
result = judge.judge("./demo.wav", rubric, mode="generate")
for it in result.items:
    print(f"  [{it.id}] {it.answer}  ({it.dimension}) {it.question}")
    print(f"        evidence: {it.evidence}")
```

| mode | inference cost | output | sample score |
|---|---|---|---|
| `logits` (default) | N forwards (one per rubric, `max_new_tokens=1`) | per-item soft `p_yes` from the next-token logits | `mean(p_yes)` — continuous, ideal as an RL reward |
| `generate` | 1 forward (long generation) | per-item `{answer, evidence}` JSON | `#yes / (#yes + #no)` — discrete, faster |

### 3. End-to-end single-audio demo

```bash
python examples/single_audio_demo.py \
    --audio /path/to/clip.wav \
    --instruction "A gentle, delicate female voice, soft and smooth pitch ..." \
    --subset speech \
    --judge_model cucl2/AnyAudio-Judge-7B \
    --mode logits
```

### 4. Reproduce benchmark numbers

The published benchmark already ships pre-computed rubric items in its
`rubric` column, so the eval script skips the (expensive) 30B decompose
and feeds them straight to the judge:

```bash
python evaluate_benchmark.py \
    --judge_model cucl2/AnyAudio-Judge-7B \
    --benchmark_repo cucl2/AnyAudio-Judge-Bench \
    --languages en zh \
    --mode logits \
    --output_dir ./eval_results
```

For `--mode generate` the script also reports per-subset accuracy after
binarizing the yes-ratio against `--yes_threshold` (default 0.8).

---

## Repository Layout

```
AnyAudio-Judge/
├── README.md
├── requirements.txt
├── anyaudio_judge/
│   ├── __init__.py
│   ├── prompts.py          # decomposition prompts (4 subsets) + judge prompts (2 modes) + cleaner prompt
│   ├── decompose.py        # Qwen3-30B-A3B-Instruct-2507 (decompose + clean) via HF transformers
│   └── judge.py            # AnyAudio-Judge audio scorer (logits / generate)
├── evaluate_benchmark.py   # benchmark eval (logits / generate, per-subset breakdown)
└── examples/
    └── single_audio_demo.py
```

---

## Citation

If you find AnyAudio-Judge helpful, please consider citing our work:

```bibtex
@misc{anyaudiojudge2026,
  title  = {AnyAudio-Judge: A Dynamic Rubric-Based Benchmark and Evaluator for Audio Instruction Following},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Preprint, under submission}
}
```

(The bibtex will be updated once the paper is officially published.)

---

## License

- The **code** in this repository is released under the Apache-2.0 license.
- The **AnyAudio-Judge Bench** and **AnyAudio-Judge Corpus** redistribute audio derived from publicly released datasets (Clotho, AudioCaps, Song Describer, MusicBench, InstructTTSEval, etc.) under their respective licenses; please consult each upstream source before commercial use.
- The **model weights** are released under the Apache-2.0 license, inheriting the license of the base Qwen2.5-Omni / Qwen3-Omni-Captioner models.

---

## Acknowledgements

This work builds on a number of excellent open-source projects, including but not limited to: Qwen-Omni, AudioGen, AudioLDM2, Stable Audio, MusicGen, ACE-Step, MOSS-VoiceGenerator, MiMo-Audio, InstructTTSEval, Clotho, AudioCaps, Song Describer, MusicBench, and PAM. We are grateful to the maintainers of these resources.
