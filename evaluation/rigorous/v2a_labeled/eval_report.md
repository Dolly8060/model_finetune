# 模型评估报告 - FTQwen3_V2A

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 34.85 |
| ROUGE-1 | 27.41 |
| ROUGE-2 | 16.22 |
| ROUGE-L | 24.94 |
| BERTScore-P | 85.41 |
| BERTScore-R | 90.65 |
| BERTScore-F1 | 87.91 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 9.57 |
| ROUGE-1 | 28.34 |
| ROUGE-2 | 9.09 |
| ROUGE-L | 20.14 |
| BERTScore-P | 83.86 |
| BERTScore-R | 85.16 |
| BERTScore-F1 | 84.48 |

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 82.12% |
| Strict Acc (完全通过率) | 56.50% |
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
| max_words | 26 | 21 | 80.8% |
| response_language | 92 | 88 | 95.7% |
| min_sentences | 68 | 52 | 76.5% |
| word_frequency | 47 | 34 | 72.3% |
| keyword_count | 35 | 24 | 68.6% |
| all_lowercase | 28 | 28 | 100.0% |
| title_double_brackets | 33 | 32 | 97.0% |
| paragraph_divider | 42 | 40 | 95.2% |
| min_words | 28 | 20 | 71.4% |
| postscript | 49 | 39 | 79.6% |
| exact_paragraphs | 24 | 20 | 83.3% |
| keyword_exclude | 5 | 5 | 100.0% |
| zh_no_degree_adverbs | 4 | 4 | 100.0% |
| highlight_sections | 29 | 25 | 86.2% |
| section_markers | 13 | 12 | 92.3% |
| placeholder_count | 31 | 22 | 71.0% |
| numbered_list | 3 | 3 | 100.0% |
| end_with_question | 4 | 1 | 25.0% |
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
| json_format | 2 | 2 | 100.0% |
| table_format | 2 | 0 | 0.0% |
| zh_no_adjectives | 1 | 1 | 100.0% |

### 内容质量参考（辅助指标）

| 指标 | 分数 |
|------|------|
| BLEU | 24.12 |
| ROUGE-1 | 37.85 |
| ROUGE-2 | 19.05 |
| ROUGE-L | 26.52 |
| BERTScore-P | 88.09 |
| BERTScore-R | 89.13 |
| BERTScore-F1 | 88.59 |