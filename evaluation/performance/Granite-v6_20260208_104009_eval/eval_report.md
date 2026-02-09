# 模型评估报告 - Granite-v6

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 82.77 |
| ROUGE-1 | 52.86 |
| ROUGE-2 | 37.16 |
| ROUGE-L | 50.21 |
| BERTScore-P | 92.66 |
| BERTScore-R | 92.73 |
| BERTScore-F1 | 92.67 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 39.38 |
| ROUGE-1 | 40.44 |
| ROUGE-2 | 18.08 |
| ROUGE-L | 35.00 |
| BERTScore-P | 87.92 |
| BERTScore-R | 87.44 |
| BERTScore-F1 | 87.67 |

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 87.70% |
| Strict Acc (完全通过率) | 71.07% |
| Loose Acc (宽松通过率) | 89.67% |

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
| response_language | 92 | 89 | 96.7% |
| min_sentences | 70 | 63 | 90.0% |
| title_double_brackets | 35 | 35 | 100.0% |
| paragraph_divider | 42 | 42 | 100.0% |
| word_frequency | 47 | 40 | 85.1% |
| exact_paragraphs | 26 | 21 | 80.8% |
| min_words | 32 | 23 | 71.9% |
| keyword_count | 35 | 27 | 77.1% |
| max_words | 26 | 25 | 96.2% |
| keyword_include_zh | 8 | 7 | 87.5% |
| highlight_sections | 30 | 28 | 93.3% |
| section_markers | 13 | 12 | 92.3% |
| zh_three_part_structure | 2 | 2 | 100.0% |
| all_lowercase | 28 | 28 | 100.0% |
| quotation_wrap | 8 | 8 | 100.0% |
| bullet_points | 5 | 1 | 20.0% |
| postscript | 55 | 44 | 80.0% |
| placeholder_count | 35 | 30 | 85.7% |
| max_sentences | 7 | 7 | 100.0% |
| start_with | 4 | 4 | 100.0% |
| end_with_question | 4 | 2 | 50.0% |
| numbered_list | 3 | 3 | 100.0% |
| zh_keyword_per_paragraph | 2 | 2 | 100.0% |
| keyword_exclude | 8 | 8 | 100.0% |
| zh_no_degree_adverbs | 4 | 4 | 100.0% |
| min_paragraphs | 4 | 4 | 100.0% |
| table_format | 14 | 0 | 0.0% |
| json_format | 6 | 6 | 100.0% |
| zh_qa_format | 4 | 4 | 100.0% |
| zh_no_adjectives | 1 | 1 | 100.0% |
| exact_words | 2 | 0 | 0.0% |
| zh_keyword_count | 2 | 2 | 100.0% |
| repeat_prompt | 8 | 8 | 100.0% |
| separator_asterisks | 6 | 6 | 100.0% |
| markdown_format | 1 | 0 | 0.0% |
| no_commas | 6 | 6 | 100.0% |

### 内容质量参考（辅助指标）（仅基于 179 个有reference的样本，1056 个无reference已跳过）

| 指标 | 分数 |
|------|------|
| BLEU | 18.01 |
| ROUGE-1 | 44.38 |
| ROUGE-2 | 24.99 |
| ROUGE-L | 32.34 |
| BERTScore-P | 90.57 |
| BERTScore-R | 90.23 |
| BERTScore-F1 | 90.38 |