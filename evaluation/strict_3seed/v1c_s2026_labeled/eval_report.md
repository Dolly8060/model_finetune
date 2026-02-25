# 模型评估报告 - FTQwen3_V1C_S2026

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 19.28 |
| ROUGE-1 | 25.66 |
| ROUGE-2 | 13.38 |
| ROUGE-L | 22.26 |
| BERTScore-P | 85.52 |
| BERTScore-R | 90.56 |
| BERTScore-F1 | 87.93 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 4.25 |
| ROUGE-1 | 30.96 |
| ROUGE-2 | 8.43 |
| ROUGE-L | 19.48 |
| BERTScore-P | 82.65 |
| BERTScore-R | 83.09 |
| BERTScore-F1 | 82.84 |

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 58.33% |
| Strict Acc (完全通过率) | 54.55% |
| Loose Acc (宽松通过率) | 54.55% |

### 统计信息

- 指令遵循总样本数: 300
- 检测到约束的样本数: 11
- 无可检测约束的样本数: 289
- 总约束数: 12
- 平均约束数/样本: 1.09
- 约束覆盖率: 3.7%

> 注: 289 个样本未检测到可验证的约束（可能是纯问答/总结任务被归入IF类，或约束模式未覆盖）。
> IFR/Strict Acc/Loose Acc 仅基于 11 个有约束的样本计算。

### 按约束类型分解

| 约束类型 | 总数 | 通过 | 通过率 |
|----------|------|------|--------|
| max_words | 3 | 2 | 66.7% |
| table_format | 7 | 3 | 42.9% |
| json_format | 1 | 1 | 100.0% |
| markdown_format | 1 | 1 | 100.0% |

### 内容质量参考（辅助指标）

| 指标 | 分数 |
|------|------|
| BLEU | 6.34 |
| ROUGE-1 | 16.40 |
| ROUGE-2 | 10.33 |
| ROUGE-L | 14.97 |
| BERTScore-P | 76.93 |
| BERTScore-R | 84.71 |
| BERTScore-F1 | 80.55 |