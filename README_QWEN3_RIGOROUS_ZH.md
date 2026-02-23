# Qwen3 Rigorous 微调与评测指南（中文版）

本指南面向 `D:/AI_code/model_finetune` 工程，目标是对 `Qwen3-1.7B` 进行三任务微调与严谨评测：
- 翻译（Translation）
- 总结（Summarization）
- 指令遵循（Instruction Following）

## 1. 设计目标

该方案强调“可复现 + 可审计 + 可对比”：
- 训练/验证/测试严格隔离
- prompt 级去重，避免数据泄漏
- 标注测试集与无标注 IF 测试集分开评估
- 推理与评分分离（`generate.py` + `score.py`）

## 2. 环境准备

```bash
conda activate granite_ft
cd /d D:/AI_code/model_finetune
```

## 3. 构建严格数据集

```bash
python scripts/build_qwen3_rigorous_dataset.py \
  --train-target 3600 \
  --val-target 360 \
  --test-target 600 \
  --if-unlabeled-target 1200 \
  --output-prefix qwen3_rigorous
```

输出文件：
- `data/qwen3_rigorous_train.json`
- `data/qwen3_rigorous_val.json`
- `data/qwen3_rigorous_test_labeled.json`
- `data/qwen3_rigorous_test_if_unlabeled.json`
- `data/qwen3_rigorous_manifest.json`

建议先检查 `qwen3_rigorous_manifest.json` 中泄漏项是否均为 0。

## 4. 训练（LoRA）

推荐配置：
- `configs/finetune_qwen3_lora_rigorous.yaml`

训练命令：

```bash
llamafactory-cli train configs/finetune_qwen3_lora_rigorous.yaml
```

默认输出目录：
- `outputs/qwen3-1.7B-lora-rigorous`

## 5. 评测流程（两步，推荐）

### 步骤 1：推理（generate）

说明：
- `--max-input-length` 建议与训练 `cutoff_len` 对齐（本方案用 2048）
- 推理结果会保留 `source` / `language` 字段，便于审计

基座模型：

```bash
python scripts/generate.py --models \
  "BaseQwen3:D:/AI_code/models/Qwen3-1.7B" \
  --eval-file data/qwen3_rigorous_test_labeled.json \
  --output-file evaluation/output_data/base_labeled.json \
  --max-input-length 2048
```

微调模型：

```bash
python scripts/generate.py --models \
  "FTQwen3:D:/AI_code/models/Qwen3-1.7B:outputs/qwen3-1.7B-lora-rigorous" \
  --eval-file data/qwen3_rigorous_test_labeled.json \
  --output-file evaluation/output_data/ft_labeled.json \
  --max-input-length 2048
```

### 步骤 2：评分（score）

```bash
python scripts/score.py --input-file evaluation/output_data/base_labeled.json --output-dir evaluation/rigorous/base_labeled
python scripts/score.py --input-file evaluation/output_data/ft_labeled.json --output-dir evaluation/rigorous/ft_labeled
```

输出文件：
- `eval_results.json`
- `eval_report.md`

## 6. 无标注 IF 基准评测

该集合主要看指令遵循指标（IFR / Strict / Loose）：

```bash
python scripts/generate.py --models "BaseQwen3:D:/AI_code/models/Qwen3-1.7B" --eval-file data/qwen3_rigorous_test_if_unlabeled.json --output-file evaluation/output_data/base_if_unlabeled.json --max-input-length 2048

python scripts/generate.py --models "FTQwen3:D:/AI_code/models/Qwen3-1.7B:outputs/qwen3-1.7B-lora-rigorous" --eval-file data/qwen3_rigorous_test_if_unlabeled.json --output-file evaluation/output_data/ft_if_unlabeled.json --max-input-length 2048

python scripts/score.py --input-file evaluation/output_data/base_if_unlabeled.json --output-dir evaluation/rigorous/base_if_unlabeled
python scripts/score.py --input-file evaluation/output_data/ft_if_unlabeled.json --output-dir evaluation/rigorous/ft_if_unlabeled
```

## 7. 评分逻辑说明（当前版本）

- `score.py` 默认保留原始 `task_type`（严格评测推荐）
- 如需旧行为（运行时重分类），手动加：`--reclassify`
- BLEU 已按语种自适应分组计算（中文与非中文分开再加权）

## 8. 推荐判定阈值（可按业务调整）

- 翻译：BLEU 提升 >= 3 或 ROUGE-L 提升 >= 4
- 总结：ROUGE-L 提升 >= 4 且 BERTScore-F1 提升 >= 1
- 指令遵循：IFR 提升 >= 8 且 Strict 提升 >= 5

## 9. 常见问题

1. 显存不足（OOM）
- 降低 `per_device_train_batch_size`
- 降低 `cutoff_len` 到 1536
- 降低 `--max-input-length`

2. 训练完成但不知道是否成功
- 看到最终 `eval metrics` 且无异常堆栈，一般表示训练成功

3. 为什么不直接用 `evaluate.py`
- `evaluate.py` 是“推理+评分一体”
- 严格实验建议两步法，结果更可复核与复用
