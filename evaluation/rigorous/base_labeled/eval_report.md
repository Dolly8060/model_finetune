# 模型评估报告 - BaseQwen3

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 3.67 |
| ROUGE-1 | 11.94 |
| ROUGE-2 | 6.39 |
| ROUGE-L | 10.32 |
| BERTScore-P | 73.16 |
| BERTScore-R | 84.19 |
| BERTScore-F1 | 78.20 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 2.03 |
| ROUGE-1 | 13.85 |
| ROUGE-2 | 3.95 |
| ROUGE-L | 9.34 |
| BERTScore-P | 73.71 |
| BERTScore-R | 78.19 |
| BERTScore-F1 | 75.80 |

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 81.95% |
| Strict Acc (完全通过率) | 51.98% |
| Loose Acc (宽松通过率) | 89.27% |

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
| max_words | 26 | 3 | 11.5% |
| response_language | 92 | 83 | 90.2% |
| min_sentences | 68 | 68 | 100.0% |
| word_frequency | 47 | 45 | 95.7% |
| keyword_count | 35 | 33 | 94.3% |
| all_lowercase | 28 | 28 | 100.0% |
| title_double_brackets | 33 | 32 | 97.0% |
| paragraph_divider | 42 | 32 | 76.2% |
| min_words | 28 | 28 | 100.0% |
| postscript | 49 | 37 | 75.5% |
| exact_paragraphs | 24 | 5 | 20.8% |
| keyword_exclude | 5 | 0 | 0.0% |
| zh_no_degree_adverbs | 4 | 0 | 0.0% |
| highlight_sections | 29 | 25 | 86.2% |
| section_markers | 13 | 12 | 92.3% |
| placeholder_count | 31 | 31 | 100.0% |
| numbered_list | 3 | 3 | 100.0% |
| end_with_question | 4 | 4 | 100.0% |
| zh_qa_format | 4 | 4 | 100.0% |
| start_with | 4 | 0 | 0.0% |
| zh_keyword_per_paragraph | 2 | 2 | 100.0% |
| quotation_wrap | 3 | 0 | 0.0% |
| keyword_include_zh | 8 | 8 | 100.0% |
| exact_words | 2 | 0 | 0.0% |
| bullet_points | 4 | 3 | 75.0% |
| max_sentences | 5 | 1 | 20.0% |
| min_paragraphs | 4 | 4 | 100.0% |
| zh_three_part_structure | 2 | 2 | 100.0% |
| json_format | 2 | 1 | 50.0% |
| table_format | 2 | 0 | 0.0% |
| zh_no_adjectives | 1 | 1 | 100.0% |

### 内容质量参考（辅助指标）

| 指标 | 分数 |
|------|------|
| BLEU | 7.42 |
| ROUGE-1 | 25.15 |
| ROUGE-2 | 9.53 |
| ROUGE-L | 14.93 |
| BERTScore-P | 82.70 |
| BERTScore-R | 85.74 |
| BERTScore-F1 | 84.17 |