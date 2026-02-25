# 模型评估报告 - PlanA_Routed_tsif_v1

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 20.38 |
| ROUGE-1 | 26.18 |
| ROUGE-2 | 13.84 |
| ROUGE-L | 23.02 |
| BERTScore-P | 85.60 |
| BERTScore-R | 90.73 |
| BERTScore-F1 | 88.05 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 4.37 |
| ROUGE-1 | 30.30 |
| ROUGE-2 | 8.59 |
| ROUGE-L | 19.07 |
| BERTScore-P | 82.48 |
| BERTScore-R | 83.15 |
| BERTScore-F1 | 82.79 |

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
| BLEU | 6.26 |
| ROUGE-1 | 16.54 |
| ROUGE-2 | 10.84 |
| ROUGE-L | 15.35 |
| BERTScore-P | 76.93 |
| BERTScore-R | 84.74 |
| BERTScore-F1 | 80.57 |