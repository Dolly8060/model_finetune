# 模型评估报告 - BaseQwen3

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 49.59% |
| Strict Acc (完全通过率) | 40.74% |
| Loose Acc (宽松通过率) | 52.59% |

### 统计信息

- 指令遵循总样本数: 1200
- 检测到约束的样本数: 270
- 无可检测约束的样本数: 930
- 总约束数: 369
- 平均约束数/样本: 1.37
- 约束覆盖率: 22.5%

> 注: 930 个样本未检测到可验证的约束（可能是纯问答/总结任务被归入IF类，或约束模式未覆盖）。
> IFR/Strict Acc/Loose Acc 仅基于 270 个有约束的样本计算。

### 按约束类型分解

| 约束类型 | 总数 | 通过 | 通过率 |
|----------|------|------|--------|
| repeat_prompt | 25 | 25 | 100.0% |
| no_commas | 14 | 0 | 0.0% |
| all_uppercase | 28 | 0 | 0.0% |
| table_format | 19 | 2 | 10.5% |
| response_language | 25 | 25 | 100.0% |
| placeholder_count | 16 | 14 | 87.5% |
| min_words | 21 | 14 | 66.7% |
| all_lowercase | 22 | 21 | 95.5% |
| json_format | 16 | 8 | 50.0% |
| exact_paragraphs | 16 | 2 | 12.5% |
| markdown_format | 13 | 4 | 30.8% |
| quotation_wrap | 21 | 0 | 0.0% |
| keyword_exclude | 5 | 0 | 0.0% |
| separator_asterisks | 20 | 8 | 40.0% |
| postscript | 14 | 12 | 85.7% |
| title_double_brackets | 23 | 19 | 82.6% |
| min_sentences | 12 | 7 | 58.3% |
| max_sentences | 16 | 0 | 0.0% |
| word_frequency | 9 | 8 | 88.9% |
| keyword_count | 6 | 3 | 50.0% |
| highlight_sections | 12 | 7 | 58.3% |
| paragraph_divider | 2 | 1 | 50.0% |
| section_markers | 2 | 1 | 50.0% |
| max_words | 10 | 1 | 10.0% |
| start_with | 1 | 0 | 0.0% |
| bullet_points | 1 | 1 | 100.0% |

### 内容质量参考（辅助指标）

> 所有 1200 个指令遵循样本均无reference输出，无法计算内容质量指标。
> 指令遵循任务的核心评估依赖上述约束检测指标（IFR/Strict Acc/Loose Acc）。