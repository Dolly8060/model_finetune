# 模型评估报告 - BaseQwen3

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 45.57% |
| Strict Acc (完全通过率) | 34.82% |
| Loose Acc (宽松通过率) | 50.47% |

### 统计信息

- 指令遵循总样本数: 1200
- 检测到约束的样本数: 1169
- 无可检测约束的样本数: 31
- 总约束数: 1918
- 平均约束数/样本: 1.64
- 约束覆盖率: 97.4%

> 注: 31 个样本未检测到可验证的约束（可能是纯问答/总结任务被归入IF类，或约束模式未覆盖）。
> IFR/Strict Acc/Loose Acc 仅基于 1169 个有约束的样本计算。

### 按约束类型分解

| 约束类型 | 总数 | 通过 | 通过率 |
|----------|------|------|--------|
| separator_asterisks | 73 | 24 | 32.9% |
| multiple_responses | 58 | 58 | 100.0% |
| no_commas | 166 | 10 | 6.0% |
| repeat_prompt | 105 | 105 | 100.0% |
| title_double_brackets | 49 | 41 | 83.7% |
| exact_paragraphs | 63 | 11 | 17.5% |
| response_language | 68 | 68 | 100.0% |
| max_words | 19 | 2 | 10.5% |
| all_uppercase | 31 | 0 | 0.0% |
| min_words | 70 | 40 | 57.1% |
| highlight_sections | 54 | 38 | 70.4% |
| markdown_format | 90 | 37 | 41.1% |
| quotation_wrap | 98 | 0 | 0.0% |
| table_format | 19 | 2 | 10.5% |
| capital_word_frequency | 18 | 13 | 72.2% |
| min_sentences | 37 | 28 | 75.7% |
| bullet_points | 31 | 24 | 77.4% |
| exact_bullet_points | 49 | 20 | 40.8% |
| postscript | 92 | 60 | 65.2% |
| placeholder_count | 63 | 54 | 85.7% |
| all_lowercase | 31 | 30 | 96.8% |
| keyword_exclude | 285 | 46 | 16.1% |
| paragraph_divider | 16 | 1 | 6.2% |
| nth_paragraph_first_word | 25 | 0 | 0.0% |
| max_sentences | 41 | 1 | 2.4% |
| letter_frequency | 25 | 14 | 56.0% |
| end_with_phrase | 81 | 10 | 12.3% |
| json_format | 32 | 19 | 59.4% |
| keyword_include | 73 | 73 | 100.0% |
| section_markers | 12 | 7 | 58.3% |
| word_frequency | 9 | 8 | 88.9% |
| keyword_count | 34 | 30 | 88.2% |
| start_with | 1 | 0 | 0.0% |

### 内容质量参考（辅助指标）

> 所有 1200 个指令遵循样本均无reference输出，无法计算内容质量指标。
> 指令遵循任务的核心评估依赖上述约束检测指标（IFR/Strict Acc/Loose Acc）。