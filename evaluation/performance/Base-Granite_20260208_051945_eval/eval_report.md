# 模型评估报告 - Base-Granite

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 71.25 |
| ROUGE-1 | 43.56 |
| ROUGE-2 | 26.15 |
| ROUGE-L | 40.89 |
| BERTScore-P | 91.07 |
| BERTScore-R | 92.00 |
| BERTScore-F1 | 91.50 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 18.96 |
| ROUGE-1 | 16.63 |
| ROUGE-2 | 6.06 |
| ROUGE-L | 13.16 |
| BERTScore-P | 77.69 |
| BERTScore-R | 81.15 |
| BERTScore-F1 | 79.29 |

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 88.89% |
| Strict Acc (完全通过率) | 72.73% |
| Loose Acc (宽松通过率) | 88.02% |

### 统计信息

- 指令遵循总样本数: 1235
- 检测到约束的样本数: 242
- 无可检测约束的样本数: 993
- 总约束数: 675
- 平均约束数/样本: 2.79
- 约束覆盖率: 19.6%

> 注: 993 个样本未检测到可验证的约束（可能是纯问答/总结任务被归入IF类，或约束模式未覆盖）。
> IFR/Strict Acc/Loose Acc 仅基于 242 个有约束的样本计算。

### 按约束类型分解

| 约束类型 | 总数 | 通过 | 通过率 |
|----------|------|------|--------|
| response_language | 92 | 88 | 95.7% |
| min_sentences | 70 | 68 | 97.1% |
| title_double_brackets | 35 | 35 | 100.0% |
| paragraph_divider | 42 | 42 | 100.0% |
| word_frequency | 47 | 41 | 87.2% |
| exact_paragraphs | 26 | 17 | 65.4% |
| min_words | 32 | 29 | 90.6% |
| keyword_count | 35 | 28 | 80.0% |
| max_words | 26 | 23 | 88.5% |
| keyword_include_zh | 8 | 8 | 100.0% |
| highlight_sections | 30 | 29 | 96.7% |
| section_markers | 13 | 13 | 100.0% |
| zh_three_part_structure | 2 | 2 | 100.0% |
| all_lowercase | 28 | 28 | 100.0% |
| quotation_wrap | 8 | 8 | 100.0% |
| bullet_points | 5 | 2 | 40.0% |
| postscript | 55 | 45 | 81.8% |
| placeholder_count | 35 | 30 | 85.7% |
| max_sentences | 7 | 7 | 100.0% |
| start_with | 4 | 4 | 100.0% |
| end_with_question | 4 | 2 | 50.0% |
| numbered_list | 3 | 3 | 100.0% |
| zh_keyword_per_paragraph | 2 | 0 | 0.0% |
| keyword_exclude | 8 | 8 | 100.0% |
| zh_no_degree_adverbs | 4 | 4 | 100.0% |
| min_paragraphs | 4 | 4 | 100.0% |
| table_format | 14 | 1 | 7.1% |
| json_format | 6 | 5 | 83.3% |
| zh_qa_format | 4 | 4 | 100.0% |
| zh_no_adjectives | 1 | 1 | 100.0% |
| exact_words | 2 | 1 | 50.0% |
| zh_keyword_count | 2 | 2 | 100.0% |
| repeat_prompt | 8 | 8 | 100.0% |
| separator_asterisks | 6 | 6 | 100.0% |
| markdown_format | 1 | 0 | 0.0% |
| no_commas | 6 | 4 | 66.7% |

### 内容质量参考（辅助指标）（仅基于 179 个有reference的样本，1056 个无reference已跳过）

| 指标 | 分数 |
|------|------|
| BLEU | 24.41 |
| ROUGE-1 | 45.12 |
| ROUGE-2 | 25.01 |
| ROUGE-L | 32.05 |
| BERTScore-P | 89.86 |
| BERTScore-R | 89.88 |
| BERTScore-F1 | 89.85 |