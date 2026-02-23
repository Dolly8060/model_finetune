# 模型评估报告 - BaseQwen3

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 50.00% |
| Strict Acc (完全通过率) | 40.66% |
| Loose Acc (宽松通过率) | 53.11% |

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
| min_sentences | 12 | 7 | 58.3% |
| quotation_wrap | 21 | 0 | 0.0% |
| highlight_sections | 9 | 5 | 55.6% |
| markdown_format | 10 | 1 | 10.0% |
| max_sentences | 14 | 0 | 0.0% |
| json_format | 13 | 6 | 46.2% |
| repeat_prompt | 23 | 23 | 100.0% |
| exact_paragraphs | 16 | 2 | 12.5% |
| no_commas | 14 | 0 | 0.0% |
| paragraph_divider | 2 | 1 | 50.0% |
| keyword_count | 7 | 5 | 71.4% |
| all_uppercase | 34 | 0 | 0.0% |
| word_frequency | 11 | 10 | 90.9% |
| min_words | 22 | 16 | 72.7% |
| start_with | 2 | 0 | 0.0% |
| response_language | 27 | 27 | 100.0% |
| all_lowercase | 28 | 26 | 92.9% |
| title_double_brackets | 23 | 19 | 82.6% |
| keyword_exclude | 4 | 0 | 0.0% |
| max_words | 11 | 1 | 9.1% |
| separator_asterisks | 19 | 8 | 42.1% |
| postscript | 17 | 14 | 82.4% |
| table_format | 19 | 2 | 10.5% |
| placeholder_count | 14 | 13 | 92.9% |
| section_markers | 2 | 1 | 50.0% |

### 内容质量参考（辅助指标）

> 所有 1200 个指令遵循样本均无reference输出，无法计算内容质量指标。
> 指令遵循任务的核心评估依赖上述约束检测指标（IFR/Strict Acc/Loose Acc）。