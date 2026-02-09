# 模型评估报告 - Qwen3-vl-4b

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 83.37 |
| ROUGE-1 | 49.27 |
| ROUGE-2 | 31.26 |
| ROUGE-L | 46.79 |
| BERTScore-P | 93.30 |
| BERTScore-R | 93.59 |
| BERTScore-F1 | 93.43 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 23.74 |
| ROUGE-1 | 20.68 |
| ROUGE-2 | 7.43 |
| ROUGE-L | 17.53 |
| BERTScore-P | 78.09 |
| BERTScore-R | 81.87 |
| BERTScore-F1 | 79.86 |

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 90.22% |
| Strict Acc (完全通过率) | 76.03% |
| Loose Acc (宽松通过率) | 90.50% |

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
| min_sentences | 70 | 67 | 95.7% |
| title_double_brackets | 35 | 35 | 100.0% |
| paragraph_divider | 42 | 40 | 95.2% |
| word_frequency | 47 | 44 | 93.6% |
| exact_paragraphs | 26 | 18 | 69.2% |
| min_words | 32 | 27 | 84.4% |
| keyword_count | 35 | 33 | 94.3% |
| max_words | 26 | 24 | 92.3% |
| keyword_include_zh | 8 | 8 | 100.0% |
| highlight_sections | 30 | 30 | 100.0% |
| section_markers | 13 | 12 | 92.3% |
| zh_three_part_structure | 2 | 2 | 100.0% |
| all_lowercase | 28 | 28 | 100.0% |
| quotation_wrap | 8 | 7 | 87.5% |
| bullet_points | 5 | 3 | 60.0% |
| postscript | 55 | 44 | 80.0% |
| placeholder_count | 35 | 33 | 94.3% |
| max_sentences | 7 | 7 | 100.0% |
| start_with | 4 | 3 | 75.0% |
| end_with_question | 4 | 4 | 100.0% |
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
| separator_asterisks | 6 | 5 | 83.3% |
| markdown_format | 1 | 0 | 0.0% |
| no_commas | 6 | 5 | 83.3% |

### 内容质量参考（辅助指标）（仅基于 179 个有reference的样本，1056 个无reference已跳过）

| 指标 | 分数 |
|------|------|
| BLEU | 20.39 |
| ROUGE-1 | 39.36 |
| ROUGE-2 | 17.43 |
| ROUGE-L | 26.15 |
| BERTScore-P | 88.01 |
| BERTScore-R | 88.98 |
| BERTScore-F1 | 88.48 |