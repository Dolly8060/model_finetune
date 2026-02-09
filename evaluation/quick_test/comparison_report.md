# 多模型对比评估报告

## 评估指标说明

- **BLEU**: 机器翻译标准评估指标，主要用于翻译子集
- **ROUGE-1/2/L**: 文本摘要评估指标，主要用于总结子集
- **BERTScore**: 基于BERT的语义相似度评估

## 模型信息

| 模型名称 | 模型路径 |
|----------|----------|
| Base-Granite-1B | D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16 |
| Qwen3-4B | D:/AI_code/models/Qwen3-4B-Instruct-2507 |

## 翻译子集 (Translation)

| 指标 | Base-Granite-1B | Qwen3-4B |
|------|-------:|-------:|
| BLEU | 30.34 | 46.65 |
| ROUGE-1 | 52.40 | 57.87 |
| ROUGE-2 | 45.97 | 51.62 |
| ROUGE-L | 52.40 | 57.87 |
| BERTScore-P | 92.87 | 95.32 |
| BERTScore-R | 96.31 | 97.95 |
| BERTScore-F1 | 94.54 | 96.61 |

### 翻译子集 (Translation) - 最佳模型

| 指标 | 最佳模型 | 分数 |
|------|----------|------|
| BLEU | Qwen3-4B | 46.65 |
| ROUGE-1 | Qwen3-4B | 57.87 |
| ROUGE-2 | Qwen3-4B | 51.62 |
| ROUGE-L | Qwen3-4B | 57.87 |
| BERTScore-P | Qwen3-4B | 95.32 |
| BERTScore-R | Qwen3-4B | 97.95 |
| BERTScore-F1 | Qwen3-4B | 96.61 |

## 总结子集 (Summarization)

| 指标 | Base-Granite-1B | Qwen3-4B |
|------|-------:|-------:|
| BLEU | 0.17 | 0.27 |
| ROUGE-1 | 1.68 | 3.48 |
| ROUGE-2 | 0.00 | 0.00 |
| ROUGE-L | 1.68 | 3.48 |
| BERTScore-P | 66.65 | 67.83 |
| BERTScore-R | 75.65 | 76.86 |
| BERTScore-F1 | 70.87 | 72.06 |

### 总结子集 (Summarization) - 最佳模型

| 指标 | 最佳模型 | 分数 |
|------|----------|------|
| BLEU | Qwen3-4B | 0.27 |
| ROUGE-1 | Qwen3-4B | 3.48 |
| ROUGE-2 | Base-Granite-1B | 0.00 |
| ROUGE-L | Qwen3-4B | 3.48 |
| BERTScore-P | Qwen3-4B | 67.83 |
| BERTScore-R | Qwen3-4B | 76.86 |
| BERTScore-F1 | Qwen3-4B | 72.06 |

## 指令遵循子集 (Instruction Following)

| 指标 | Base-Granite-1B | Qwen3-4B |
|------|-------:|-------:|
| IFR (约束通过率) | 100.00% | 100.00% |
| Strict Acc (完全通过率) | 100.00% | 100.00% |
| Loose Acc (宽松通过率) | 100.00% | 100.00% |

### 指令遵循子集 (Instruction Following) - 最佳模型

| 指标 | 最佳模型 | 分数 |
|------|----------|------|
| IFR (约束通过率) | Base-Granite-1B | 100.00% |
| Strict Acc (完全通过率) | Base-Granite-1B | 100.00% |
| Loose Acc (宽松通过率) | Base-Granite-1B | 100.00% |

## 综合结论


**翻译子集 (Translation)综合排名**：

1. **Qwen3-4B**: 平均得分 71.99
2. **Base-Granite-1B**: 平均得分 66.40

**总结子集 (Summarization)综合排名**：

1. **Qwen3-4B**: 平均得分 32.00
2. **Base-Granite-1B**: 平均得分 30.96

**指令遵循子集 (Instruction Following)综合排名**：

1. **Base-Granite-1B**: 平均得分 100.00
2. **Qwen3-4B**: 平均得分 100.00