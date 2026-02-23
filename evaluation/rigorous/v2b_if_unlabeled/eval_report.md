# 模型评估报告 - FTQwen3_V2B

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 66.04% |
| Strict Acc (完全通过率) | 58.24% |
| Loose Acc (宽松通过率) | 69.96% |

### 统计信息

- 指令遵循总样本数: 1200
- 检测到约束的样本数: 273
- 无可检测约束的样本数: 927
- 总约束数: 374
- 平均约束数/样本: 1.37
- 约束覆盖率: 22.8%

> 注: 927 个样本未检测到可验证的约束（可能是纯问答/总结任务被归入IF类，或约束模式未覆盖）。
> IFR/Strict Acc/Loose Acc 仅基于 273 个有约束的样本计算。

### 按约束类型分解

| 约束类型 | 总数 | 通过 | 通过率 |
|----------|------|------|--------|
| min_sentences | 12 | 4 | 33.3% |
| quotation_wrap | 21 | 0 | 0.0% |
| highlight_sections | 9 | 6 | 66.7% |
| markdown_format | 10 | 4 | 40.0% |
| max_sentences | 14 | 12 | 85.7% |
| json_format | 13 | 13 | 100.0% |
| repeat_prompt | 23 | 23 | 100.0% |
| exact_paragraphs | 16 | 14 | 87.5% |
| no_commas | 14 | 5 | 35.7% |
| paragraph_divider | 2 | 2 | 100.0% |
| keyword_count | 7 | 3 | 42.9% |
| all_uppercase | 34 | 18 | 52.9% |
| word_frequency | 11 | 6 | 54.5% |
| min_words | 22 | 15 | 68.2% |
| start_with | 2 | 0 | 0.0% |
| response_language | 27 | 27 | 100.0% |
| all_lowercase | 28 | 26 | 92.9% |
| title_double_brackets | 23 | 22 | 95.7% |
| keyword_exclude | 4 | 2 | 50.0% |
| max_words | 11 | 10 | 90.9% |
| separator_asterisks | 19 | 10 | 52.6% |
| postscript | 17 | 14 | 82.4% |
| table_format | 19 | 4 | 21.1% |
| placeholder_count | 14 | 6 | 42.9% |
| section_markers | 2 | 1 | 50.0% |

### 内容质量参考（辅助指标）

> 所有 1200 个指令遵循样本均无reference输出，无法计算内容质量指标。
> 指令遵循任务的核心评估依赖上述约束检测指标（IFR/Strict Acc/Loose Acc）。