# 模型评估报告 - FTQwen3

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 31.68 |
| ROUGE-1 | 27.16 |
| ROUGE-2 | 16.35 |
| ROUGE-L | 24.77 |
| BERTScore-P | 85.35 |
| BERTScore-R | 90.63 |
| BERTScore-F1 | 87.87 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 9.19 |
| ROUGE-1 | 26.91 |
| ROUGE-2 | 8.03 |
| ROUGE-L | 19.46 |
| BERTScore-P | 83.44 |
| BERTScore-R | 84.99 |
| BERTScore-F1 | 84.18 |

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 85.10% |
| Strict Acc (完全通过率) | 62.15% |
| Loose Acc (宽松通过率) | 90.96% |

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
| max_words | 26 | 20 | 76.9% |
| response_language | 92 | 87 | 94.6% |
| min_sentences | 68 | 56 | 82.4% |
| word_frequency | 47 | 39 | 83.0% |
| keyword_count | 35 | 28 | 80.0% |
| all_lowercase | 28 | 28 | 100.0% |
| title_double_brackets | 33 | 33 | 100.0% |
| paragraph_divider | 42 | 39 | 92.9% |
| min_words | 28 | 22 | 78.6% |
| postscript | 49 | 41 | 83.7% |
| exact_paragraphs | 24 | 21 | 87.5% |
| keyword_exclude | 5 | 5 | 100.0% |
| zh_no_degree_adverbs | 4 | 4 | 100.0% |
| highlight_sections | 29 | 26 | 89.7% |
| section_markers | 13 | 12 | 92.3% |
| placeholder_count | 31 | 23 | 74.2% |
| numbered_list | 3 | 3 | 100.0% |
| end_with_question | 4 | 3 | 75.0% |
| zh_qa_format | 4 | 4 | 100.0% |
| start_with | 4 | 0 | 0.0% |
| zh_keyword_per_paragraph | 2 | 2 | 100.0% |
| quotation_wrap | 3 | 0 | 0.0% |
| keyword_include_zh | 8 | 7 | 87.5% |
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
| BLEU | 23.48 |
| ROUGE-1 | 38.15 |
| ROUGE-2 | 18.67 |
| ROUGE-L | 26.53 |
| BERTScore-P | 87.98 |
| BERTScore-R | 89.21 |
| BERTScore-F1 | 88.57 |