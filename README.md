# [EMNLP 2026] AnyAudio-Judge

> **A Dynamic Rubric-Based Benchmark and Evaluator for Audio Instruction Following**

[![arXiv](https://img.shields.io/badge/%F0%9F%93%84%20ArXiv-Paper-red.svg)](https://arxiv.org/abs/2606.03116)
[![github](https://img.shields.io/badge/Code-Repo-black?logo=github)](https://github.com/CuCl-2/AnyAudio-Judge)
[![Bench](https://img.shields.io/badge/%F0%9F%A4%97%20Bench-Dataset-blue)](https://huggingface.co/datasets/cucl2/AnyAudio-Judge-Bench)
[![Corpus](https://img.shields.io/badge/%F0%9F%A4%97%20Corpus-Dataset-blue)](https://huggingface.co/datasets/cucl2/AnyAudio-Judge-Corpus)
[![Corpus-opus](https://img.shields.io/badge/%F0%9F%A4%97%20Corpus--opus-NEW-orange)](https://huggingface.co/datasets/cucl2/AnyAudio-Judge-Corpus-opus)
[![Model-7B](https://img.shields.io/badge/%F0%9F%A4%97%20Model-7B-blueviolet)](https://huggingface.co/cucl2/AnyAudio-Judge-7B)
[![Model-30B](https://img.shields.io/badge/%F0%9F%A4%97%20Model-30B-blueviolet)](https://huggingface.co/cucl2/AnyAudio-Judge-30B)

## 🔥 News

- **2026-11** — We release **AnyAudio-Judge-Corpus-opus**, a rebuilt training corpus in which rubrics and hard negatives are written by **Claude Opus 5.5**, substantially improving data quality.
  - Dataset: [`cucl2/AnyAudio-Judge-Corpus-opus`](https://huggingface.co/datasets/cucl2/AnyAudio-Judge-Corpus-opus)
  - Online viewer: [`cucl2/AnyAudio-Judge-Corpus-opus-viewer`](https://huggingface.co/spaces/cucl2/AnyAudio-Judge-Corpus-opus-viewer)
  - Construction specs: [`docs/`](https://huggingface.co/datasets/cucl2/AnyAudio-Judge-Corpus-opus/tree/main/docs)

## Overview

AnyAudio-Judge evaluates instruction-guided audio generation (speech / sound / music / mixed audio) with **dynamic rubrics**: an instruction is decomposed into a few verifiable yes/no items, an audio-language judge answers each item, and the item-level probabilities are averaged into an interpretable alignment score.

## Resources

| Resource | HuggingFace |
|---|---|
| Benchmark (en + zh) | [`cucl2/AnyAudio-Judge-Bench`](https://huggingface.co/datasets/cucl2/AnyAudio-Judge-Bench) |
| Corpus (original, 105K) | [`cucl2/AnyAudio-Judge-Corpus`](https://huggingface.co/datasets/cucl2/AnyAudio-Judge-Corpus) |
| **Corpus-opus (new)** | [`cucl2/AnyAudio-Judge-Corpus-opus`](https://huggingface.co/datasets/cucl2/AnyAudio-Judge-Corpus-opus) · [viewer](https://huggingface.co/spaces/cucl2/AnyAudio-Judge-Corpus-opus-viewer) |
| Model (7B) | [`cucl2/AnyAudio-Judge-7B`](https://huggingface.co/cucl2/AnyAudio-Judge-7B) |
| Model (30B) | [`cucl2/AnyAudio-Judge-30B`](https://huggingface.co/cucl2/AnyAudio-Judge-30B) |

## Quick Start

```bash
git clone https://github.com/CuCl-2/AnyAudio-Judge.git
cd AnyAudio-Judge
pip install -r requirements.txt
```

```python
from anyaudio_judge import decompose_instruction, AnyAudioJudge

rubric = decompose_instruction("A gentle, delicate female voice, calm and restrained.", subset="speech")
judge = AnyAudioJudge.from_pretrained("cucl2/AnyAudio-Judge-7B")
result = judge.judge("./demo.wav", rubric, mode="logits")
print(result.score)
```

Reproduce benchmark numbers:

```bash
python evaluate_benchmark.py --judge_model cucl2/AnyAudio-Judge-7B \
    --benchmark_repo cucl2/AnyAudio-Judge-Bench --languages en zh --mode logits
```

## Citation

```bibtex
@article{li2026anyaudio,
  title={AnyAudio-Judge: A Dynamic Rubric-Based Benchmark and Evaluator for Audio Instruction Following},
  author={Li, Haitao and Tan, Tian and Yang, Yuguang and Yang, Shan and Chen, Xie},
  journal={arXiv preprint arXiv:2606.03116},
  year={2026}
}
```

## License

Code and model weights: Apache-2.0. Datasets redistribute audio from public sources (Clotho, AudioCaps, Song Describer, MusicBench, InstructTTSEval, etc.) under their respective licenses; please check them before commercial use.
