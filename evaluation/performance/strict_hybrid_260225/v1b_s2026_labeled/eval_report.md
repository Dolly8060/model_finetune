# 模型评估报告 - FTQwen3_V1B_S2026

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 20.43 |
| ROUGE-1 | 25.65 |
| ROUGE-2 | 13.45 |
| ROUGE-L | 22.20 |
| BERTScore-P | 85.61 |
| BERTScore-R | 90.69 |
| BERTScore-F1 | 88.04 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 3.75 |
| ROUGE-1 | 28.29 |
| ROUGE-2 | 7.91 |
| ROUGE-L | 18.30 |
| BERTScore-P | 81.86 |
| BERTScore-R | 83.05 |
| BERTScore-F1 | 82.43 |

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 14.36% |
| Strict Acc (完全通过率) | 13.66% |
| Loose Acc (宽松通过率) | 14.21% |

### 统计信息

- 指令遵循总样本数: 300
- 检测到约束的样本数: 183
- 无可检测约束的样本数: 117
- 总约束数: 188
- 平均约束数/样本: 1.03
- 约束覆盖率: 61.0%

> 注: 117 个样本未检测到可验证的约束（可能是纯问答/总结任务被归入IF类，或约束模式未覆盖）。
> IFR/Strict Acc/Loose Acc 仅基于 183 个有约束的样本计算。

### 按约束类型分解

| 约束类型 | 总数 | 通过 | 通过率 |
|----------|------|------|--------|
| lria_reference_exact_short | 92 | 0 | 0.0% |
| lria_language_or_code | 28 | 18 | 64.3% |
| reply_only_choices | 52 | 0 | 0.0% |
| first_letter_lowercase | 2 | 2 | 100.0% |
| max_words | 3 | 1 | 33.3% |
| table_format | 7 | 3 | 42.9% |
| markdown_format | 3 | 2 | 66.7% |
| json_format | 1 | 1 | 100.0% |

### 内容质量参考（辅助指标）

| 指标 | 分数 |
|------|------|
| BLEU | 6.58 |
| ROUGE-1 | 16.64 |
| ROUGE-2 | 10.68 |
| ROUGE-L | 15.46 |
| BERTScore-P | 76.97 |
| BERTScore-R | 84.78 |
| BERTScore-F1 | 80.61 |