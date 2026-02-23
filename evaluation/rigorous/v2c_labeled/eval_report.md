# 模型评估报告 - FTQwen3_V2C

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 31.77 |
| ROUGE-1 | 26.94 |
| ROUGE-2 | 16.12 |
| ROUGE-L | 24.49 |
| BERTScore-P | 85.39 |
| BERTScore-R | 90.59 |
| BERTScore-F1 | 87.87 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 9.58 |
| ROUGE-1 | 28.20 |
| ROUGE-2 | 8.25 |
| ROUGE-L | 19.66 |
| BERTScore-P | 84.01 |
| BERTScore-R | 85.09 |
| BERTScore-F1 | 84.53 |

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 84.27% |
| Strict Acc (完全通过率) | 59.89% |
| Loose Acc (宽松通过率) | 93.22% |

### 统计信息

- 指令遵循总样本数: 185
- 检测到约束的样本数: 177
- 无可检测约束的样本数: 8
- 总约束数: 604
- 平均约束数/样本: 3.41
- 约束覆盖率: 95.7%

> 注: 8 个样本未检测到可验证的约束（可能是纯问答/总结任务被归入IF类，或约束模式未覆盖）。
> IFR/Strict Acc/Loose Acc 仅基于 177 个有约束的样本计算。

### 按约束类型分解

| 约束类型 | 总数 | 通过 | 通过率 |
|----------|------|------|--------|
| max_words | 26 | 22 | 84.6% |
| response_language | 92 | 88 | 95.7% |
| min_sentences | 68 | 57 | 83.8% |
| word_frequency | 47 | 37 | 78.7% |
| keyword_count | 35 | 26 | 74.3% |
| all_lowercase | 28 | 28 | 100.0% |
| title_double_brackets | 33 | 32 | 97.0% |
| paragraph_divider | 42 | 42 | 100.0% |
| min_words | 28 | 18 | 64.3% |
| postscript | 49 | 40 | 81.6% |
| exact_paragraphs | 24 | 19 | 79.2% |
| keyword_exclude | 5 | 5 | 100.0% |
| zh_no_degree_adverbs | 4 | 4 | 100.0% |
| highlight_sections | 29 | 26 | 89.7% |
| section_markers | 13 | 12 | 92.3% |
| placeholder_count | 31 | 22 | 71.0% |
| numbered_list | 3 | 3 | 100.0% |
| end_with_question | 4 | 3 | 75.0% |
| zh_qa_format | 4 | 4 | 100.0% |
| start_with | 4 | 0 | 0.0% |
| zh_keyword_per_paragraph | 2 | 2 | 100.0% |
| quotation_wrap | 3 | 0 | 0.0% |
| keyword_include_zh | 8 | 8 | 100.0% |
| exact_words | 2 | 0 | 0.0% |
| bullet_points | 4 | 0 | 0.0% |
| max_sentences | 5 | 4 | 80.0% |
| min_paragraphs | 4 | 4 | 100.0% |
| zh_three_part_structure | 2 | 2 | 100.0% |
| json_format | 2 | 0 | 0.0% |
| table_format | 2 | 0 | 0.0% |
| zh_no_adjectives | 1 | 1 | 100.0% |

### 内容质量参考（辅助指标）

| 指标 | 分数 |
|------|------|
| BLEU | 23.90 |
| ROUGE-1 | 38.36 |
| ROUGE-2 | 19.05 |
| ROUGE-L | 26.19 |
| BERTScore-P | 88.09 |
| BERTScore-R | 89.29 |
| BERTScore-F1 | 88.66 |