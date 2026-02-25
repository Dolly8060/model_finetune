# 模型评估报告 - FTQwen3_V1B_S2026

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 62.93% |
| Strict Acc (完全通过率) | 50.38% |
| Loose Acc (宽松通过率) | 65.53% |

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
| separator_asterisks | 73 | 48 | 65.8% |
| multiple_responses | 58 | 58 | 100.0% |
| no_commas | 166 | 101 | 60.8% |
| repeat_prompt | 105 | 105 | 100.0% |
| title_double_brackets | 49 | 49 | 100.0% |
| exact_paragraphs | 63 | 34 | 54.0% |
| response_language | 68 | 68 | 100.0% |
| max_words | 19 | 16 | 84.2% |
| all_uppercase | 31 | 10 | 32.3% |
| min_words | 70 | 23 | 32.9% |
| highlight_sections | 54 | 48 | 88.9% |
| markdown_format | 90 | 39 | 43.3% |
| quotation_wrap | 98 | 0 | 0.0% |
| table_format | 19 | 4 | 21.1% |
| capital_word_frequency | 18 | 11 | 61.1% |
| min_sentences | 37 | 11 | 29.7% |
| bullet_points | 31 | 27 | 87.1% |
| exact_bullet_points | 49 | 3 | 6.1% |
| postscript | 92 | 52 | 56.5% |
| placeholder_count | 63 | 37 | 58.7% |
| all_lowercase | 31 | 30 | 96.8% |
| keyword_exclude | 285 | 240 | 84.2% |
| paragraph_divider | 16 | 2 | 12.5% |
| nth_paragraph_first_word | 25 | 4 | 16.0% |
| max_sentences | 41 | 37 | 90.2% |
| letter_frequency | 25 | 12 | 48.0% |
| end_with_phrase | 81 | 16 | 19.8% |
| json_format | 32 | 31 | 96.9% |
| keyword_include | 73 | 59 | 80.8% |
| section_markers | 12 | 10 | 83.3% |
| word_frequency | 9 | 5 | 55.6% |
| keyword_count | 34 | 17 | 50.0% |
| start_with | 1 | 0 | 0.0% |

### 内容质量参考（辅助指标）

> 所有 1200 个指令遵循样本均无reference输出，无法计算内容质量指标。
> 指令遵循任务的核心评估依赖上述约束检测指标（IFR/Strict Acc/Loose Acc）。