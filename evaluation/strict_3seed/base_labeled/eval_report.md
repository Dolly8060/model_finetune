# 模型评估报告 - BaseQwen3

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 2.66 |
| ROUGE-1 | 10.66 |
| ROUGE-2 | 5.35 |
| ROUGE-L | 9.10 |
| BERTScore-P | 73.23 |
| BERTScore-R | 84.54 |
| BERTScore-F1 | 78.41 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 1.60 |
| ROUGE-1 | 18.89 |
| ROUGE-2 | 4.36 |
| ROUGE-L | 10.53 |
| BERTScore-P | 74.38 |
| BERTScore-R | 78.68 |
| BERTScore-F1 | 76.41 |

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 33.33% |
| Strict Acc (完全通过率) | 27.27% |
| Loose Acc (宽松通过率) | 36.36% |

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
| max_words | 3 | 0 | 0.0% |
| table_format | 7 | 3 | 42.9% |
| json_format | 1 | 1 | 100.0% |
| markdown_format | 1 | 0 | 0.0% |

### 内容质量参考（辅助指标）

| 指标 | 分数 |
|------|------|
| BLEU | 1.54 |
| ROUGE-1 | 7.47 |
| ROUGE-2 | 4.37 |
| ROUGE-L | 7.34 |
| BERTScore-P | 72.16 |
| BERTScore-R | 83.57 |
| BERTScore-F1 | 77.39 |