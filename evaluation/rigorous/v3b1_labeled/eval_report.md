# 模型评估报告 - FTQwen3_V3B1

## 翻译子集 (Translation)

| 指标 | 分数 |
|------|------|
| BLEU | 31.58 |
| ROUGE-1 | 26.64 |
| ROUGE-2 | 15.79 |
| ROUGE-L | 24.48 |
| BERTScore-P | 85.23 |
| BERTScore-R | 90.57 |
| BERTScore-F1 | 87.78 |

## 总结子集 (Summarization)

| 指标 | 分数 |
|------|------|
| BLEU | 9.06 |
| ROUGE-1 | 24.38 |
| ROUGE-2 | 7.56 |
| ROUGE-L | 18.08 |
| BERTScore-P | 82.79 |
| BERTScore-R | 84.82 |
| BERTScore-F1 | 83.77 |

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 85.10% |
| Strict Acc (完全通过率) | 62.15% |
| Loose Acc (宽松通过率) | 92.09% |

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
| min_sentences | 68 | 60 | 88.2% |
| word_frequency | 47 | 38 | 80.9% |
| keyword_count | 35 | 27 | 77.1% |
| all_lowercase | 28 | 28 | 100.0% |
| title_double_brackets | 33 | 32 | 97.0% |
| paragraph_divider | 42 | 42 | 100.0% |
| min_words | 28 | 20 | 71.4% |
| postscript | 49 | 41 | 83.7% |
| exact_paragraphs | 24 | 20 | 83.3% |
| keyword_exclude | 5 | 5 | 100.0% |
| zh_no_degree_adverbs | 4 | 4 | 100.0% |
| highlight_sections | 29 | 24 | 82.8% |
| section_markers | 13 | 12 | 92.3% |
| placeholder_count | 31 | 21 | 67.7% |
| numbered_list | 3 | 3 | 100.0% |
| end_with_question | 4 | 3 | 75.0% |
| zh_qa_format | 4 | 4 | 100.0% |
| start_with | 4 | 0 | 0.0% |
| zh_keyword_per_paragraph | 2 | 2 | 100.0% |
| quotation_wrap | 3 | 0 | 0.0% |
| keyword_include_zh | 8 | 6 | 75.0% |
| exact_words | 2 | 0 | 0.0% |
| bullet_points | 4 | 0 | 0.0% |
| max_sentences | 5 | 5 | 100.0% |
| min_paragraphs | 4 | 4 | 100.0% |
| zh_three_part_structure | 2 | 2 | 100.0% |
| json_format | 2 | 1 | 50.0% |
| table_format | 2 | 0 | 0.0% |
| zh_no_adjectives | 1 | 1 | 100.0% |

### 内容质量参考（辅助指标）

| 指标 | 分数 |
|------|------|
| BLEU | 23.76 |
| ROUGE-1 | 37.96 |
| ROUGE-2 | 18.68 |
| ROUGE-L | 25.95 |
| BERTScore-P | 88.12 |
| BERTScore-R | 89.27 |
| BERTScore-F1 | 88.67 |