# Qwen3-1.7B Rigorous Finetune Protocol

This document defines a reproducible workflow to prove finetuning effectiveness on:
- translation
- summarization
- instruction following

## 1. Why this protocol

The previous Qwen3 workflow could weaken benchmark credibility if public evaluation data is mixed into training.

This protocol enforces:
- strict train/val/test isolation
- prompt-level deduplication
- separate prompt-only IF benchmark (no gold answers)
- source-level traceability

## 2. Recommended public datasets (primary sources)

- OPUS-100 (translation): https://huggingface.co/datasets/Helsinki-NLP/opus-100
- WMT19 zh-en (translation benchmark): https://huggingface.co/datasets/wmt/wmt19
- arXiv Summarization (technical summarization): https://huggingface.co/datasets/ccdv/arxiv-summarization
- IFEval-like with completions (IF training): https://huggingface.co/datasets/argilla/ifeval-like-data
- IFEval prompts (IF benchmark): https://huggingface.co/datasets/HuggingFaceH4/ifeval

## 3. Build rigorous splits

Run in your conda env (`granite_ft`):

```bash
python scripts/build_qwen3_rigorous_dataset.py \
  --train-target 3600 \
  --val-target 360 \
  --test-target 600 \
  --if-unlabeled-target 1200 \
  --output-prefix qwen3_rigorous
```

Generated files:
- `data/qwen3_rigorous_train.json`
- `data/qwen3_rigorous_val.json`
- `data/qwen3_rigorous_test_labeled.json`
- `data/qwen3_rigorous_test_if_unlabeled.json`
- `data/qwen3_rigorous_manifest.json`

Use `qwen3_rigorous_manifest.json` as your audit artifact (counts + leakage check).

## 4. Train

```bash
llamafactory-cli train configs/finetune_qwen3_lora_rigorous.yaml
```

## 5. Evaluation design (two-step, required)

This project now follows strict two-step evaluation:
1. `generate.py` for inference only
2. `score.py` for scoring only

### A. Labeled test set (Base vs FT)

```bash
python scripts/generate.py --models \
  "BaseQwen3:D:/AI_code/models/Qwen3-1.7B" \
  --eval-file data/qwen3_rigorous_test_labeled.json \
  --output-file evaluation/output_data/base_labeled.json \
  --max-input-length 2048

python scripts/generate.py --models \
  "FTQwen3:D:/AI_code/models/Qwen3-1.7B:outputs/qwen3-1.7B-lora-rigorous" \
  --eval-file data/qwen3_rigorous_test_labeled.json \
  --output-file evaluation/output_data/ft_labeled.json \
  --max-input-length 2048
```

Score:

```bash
python scripts/score.py --input-file evaluation/output_data/base_labeled.json --output-dir evaluation/rigorous/base_labeled
python scripts/score.py --input-file evaluation/output_data/ft_labeled.json --output-dir evaluation/rigorous/ft_labeled
```

### B. Prompt-only IF benchmark (IFR-focused)

```bash
python scripts/generate.py --models \
  "BaseQwen3:D:/AI_code/models/Qwen3-1.7B" \
  --eval-file data/qwen3_rigorous_test_if_unlabeled.json \
  --output-file evaluation/output_data/base_if_unlabeled.json \
  --max-input-length 2048

python scripts/generate.py --models \
  "FTQwen3:D:/AI_code/models/Qwen3-1.7B:outputs/qwen3-1.7B-lora-rigorous" \
  --eval-file data/qwen3_rigorous_test_if_unlabeled.json \
  --output-file evaluation/output_data/ft_if_unlabeled.json \
  --max-input-length 2048
```

Score:

```bash
python scripts/score.py --input-file evaluation/output_data/base_if_unlabeled.json --output-dir evaluation/rigorous/base_if_unlabeled
python scripts/score.py --input-file evaluation/output_data/ft_if_unlabeled.json --output-dir evaluation/rigorous/ft_if_unlabeled
```

### C. Important scoring notes

- `score.py` now keeps original `task_type` by default (recommended for rigorous sets).
- Optional fallback (legacy behavior): add `--reclassify`.
- BLEU is computed with language-aware tokenization (Chinese and non-Chinese split).

## 6. What counts as evidence

Minimum evidence package:
- zero leakage in `qwen3_rigorous_manifest.json`
- side-by-side Base vs FT reports on labeled test
- IF metrics on prompt-only IF benchmark
- per-task gains (not only overall average)

Recommended acceptance thresholds (adjust to your business constraints):
- Translation: BLEU +3.0 or ROUGE-L +4.0
- Summarization: ROUGE-L +4.0 and BERTScore-F1 +1.0
- Instruction following: IFR +8.0 and Strict Accuracy +5.0

## 7. Stronger rigor (optional)

Run 3 seeds and report mean/std:
- seed: 2026, 2027, 2028

Compare:
- Base (no finetune)
- FT (full 3-task mix)
- Ablation FT (remove IF or reduce IF ratio)

If gains stay consistent across seeds and both IF settings, the claim is substantially stronger.
