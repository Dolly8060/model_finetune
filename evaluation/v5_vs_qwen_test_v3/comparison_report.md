# 多模型对比评估报告

## 评估指标说明

- **BLEU**: 机器翻译标准评估指标，主要用于翻译子集
- **ROUGE-1/2/L**: 文本摘要评估指标，主要用于总结子集
- **BERTScore**: 基于BERT的语义相似度评估

## 模型信息

| 模型名称 | 模型路径 |
|----------|----------|
| Qwen3-4B | D:/AI_code/models/Qwen3-4B-Instruct-2507 |
| v5微调Granite-1B | D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16 |

## 翻译子集 (Translation)

| 指标 | Qwen3-4B | v5微调Granite-1B |
|------|-------:|-------:|
| BLEU | 0.00 | 0.00 |
| ROUGE-1 | 52.56 | 57.14 |
| ROUGE-2 | 33.49 | 40.07 |
| ROUGE-L | 49.28 | 53.91 |
| BERTScore-P | 94.12 | 93.80 |
| BERTScore-R | 94.26 | 93.82 |
| BERTScore-F1 | 94.19 | 93.80 |

### 翻译子集 (Translation) - 最佳模型

| 指标 | 最佳模型 | 分数 |
|------|----------|------|
| BLEU | Qwen3-4B | 0.00 |
| ROUGE-1 | v5微调Granite-1B | 57.14 |
| ROUGE-2 | v5微调Granite-1B | 40.07 |
| ROUGE-L | v5微调Granite-1B | 53.91 |
| BERTScore-P | Qwen3-4B | 94.12 |
| BERTScore-R | Qwen3-4B | 94.26 |
| BERTScore-F1 | Qwen3-4B | 94.19 |

## 总结子集 (Summarization)

| 指标 | Qwen3-4B | v5微调Granite-1B |
|------|-------:|-------:|
| BLEU | 2.16 | 0.00 |
| ROUGE-1 | 21.22 | 34.91 |
| ROUGE-2 | 7.90 | 14.06 |
| ROUGE-L | 19.16 | 30.09 |
| BERTScore-P | 77.69 | 87.08 |
| BERTScore-R | 82.53 | 87.23 |
| BERTScore-F1 | 79.98 | 87.12 |

### 总结子集 (Summarization) - 最佳模型

| 指标 | 最佳模型 | 分数 |
|------|----------|------|
| BLEU | Qwen3-4B | 2.16 |
| ROUGE-1 | v5微调Granite-1B | 34.91 |
| ROUGE-2 | v5微调Granite-1B | 14.06 |
| ROUGE-L | v5微调Granite-1B | 30.09 |
| BERTScore-P | v5微调Granite-1B | 87.08 |
| BERTScore-R | v5微调Granite-1B | 87.23 |
| BERTScore-F1 | v5微调Granite-1B | 87.12 |

## 综合结论


**翻译子集 (Translation)综合排名**：

1. **v5微调Granite-1B**: 平均得分 61.79
2. **Qwen3-4B**: 平均得分 59.70

**总结子集 (Summarization)综合排名**：

1. **v5微调Granite-1B**: 平均得分 48.64
2. **Qwen3-4B**: 平均得分 41.52