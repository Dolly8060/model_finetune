# 模型评估报告 - FTQwen3_TSIF_V1_IF

## 指令遵循子集 (Instruction Following)

### 核心指标

| 指标 | 分数 |
|------|------|
| IFR (约束通过率) | 66.40% |
| Strict Acc (完全通过率) | 60.74% |
| Loose Acc (宽松通过率) | 69.26% |

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
| no_commas | 14 | 8 | 57.1% |
| all_uppercase | 28 | 12 | 42.9% |
| table_format | 19 | 4 | 21.1% |
| response_language | 25 | 25 | 100.0% |
| placeholder_count | 16 | 10 | 62.5% |
| min_words | 21 | 12 | 57.1% |
| all_lowercase | 22 | 21 | 95.5% |
| json_format | 16 | 15 | 93.8% |
| exact_paragraphs | 16 | 11 | 68.8% |
| markdown_format | 13 | 3 | 23.1% |
| quotation_wrap | 21 | 0 | 0.0% |
| keyword_exclude | 5 | 1 | 20.0% |
| separator_asterisks | 20 | 18 | 90.0% |
| postscript | 14 | 13 | 92.9% |
| title_double_brackets | 23 | 23 | 100.0% |
| min_sentences | 12 | 4 | 33.3% |
| max_sentences | 16 | 11 | 68.8% |
| word_frequency | 9 | 4 | 44.4% |
| keyword_count | 6 | 3 | 50.0% |
| highlight_sections | 12 | 10 | 83.3% |
| paragraph_divider | 2 | 1 | 50.0% |
| section_markers | 2 | 1 | 50.0% |
| max_words | 10 | 9 | 90.0% |
| start_with | 1 | 0 | 0.0% |
| bullet_points | 1 | 1 | 100.0% |

### 内容质量参考（辅助指标）

> 所有 1200 个指令遵循样本均无reference输出，无法计算内容质量指标。
> 指令遵循任务的核心评估依赖上述约束检测指标（IFR/Strict Acc/Loose Acc）。