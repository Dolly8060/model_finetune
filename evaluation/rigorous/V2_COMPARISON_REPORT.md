# Qwen3-1.7B V2 版本对比报告（V2A / V2B / V2C）

## 1. 报告目的
- 对比 `v2a / v2b / v2c` 三个 V2 方案相对 `V1（ft）` 的效果变化。
- 给出明确结论：哪个版本适合作为 V2 主候选，哪个版本适合作为保守备选。

## 2. 评测范围与口径
- 有标注严格集（labeled）：
  - `evaluation/rigorous/v2a_labeled/eval_results.json`
  - `evaluation/rigorous/v2b_labeled/eval_results.json`
  - `evaluation/rigorous/v2c_labeled/eval_results.json`
- 无标注 IF 集（if_unlabeled）：
  - `evaluation/rigorous/v2a_if_unlabeled/eval_results.json`
  - `evaluation/rigorous/v2b_if_unlabeled/eval_results.json`
  - `evaluation/rigorous/v2c_if_unlabeled/eval_results.json`
- 对照版本：
  - `V1`: `evaluation/rigorous/ft_labeled/eval_results.json` + `evaluation/rigorous/ft_if_unlabeled/eval_results.json`
  - `Base`: `evaluation/rigorous/base_labeled/eval_results.json` + `evaluation/rigorous/base_if_unlabeled/eval_results.json`

说明：
- 所有 V2 版本训练均完成到 `epoch=3.0`（`global_step=675`）。
- `if_unlabeled` 的约束覆盖率仍为约 `22.8%`（273 / 1200），结论适用于“可检测约束子集”。

## 3. 最终结论（管理层版）
- `v2a`：不推荐。翻译/总结略有收益，但 IF 在 labeled 和 unlabeled 两套评测上均退化。
- `v2b`：推荐作为 **V2 主候选（IF 优先）**。IF 在 labeled 和 unlabeled 两套评测上均提升，代价是翻译/总结相对 V1 小幅回退。
- `v2c`：推荐作为 **V2 保守备选（平衡版）**。翻译/总结更稳，unlabeled IF 提升明显，但 labeled IF（IFR/Strict）较 V1 略退。

## 4. 核心指标对比（Labeled）

### 4.1 绝对分数（Base / V1 / V2A / V2B / V2C）
| 版本 | 翻译 BLEU | 翻译 ROUGE-L | 总结 ROUGE-L | 总结 BERT-F1 | IF IFR | IF Strict | IF Loose |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 3.67 | 10.32 | 9.34 | 75.80 | 81.95 | 51.98 | 89.27 |
| V1 (ft) | 31.68 | 24.77 | 19.46 | 84.18 | 85.10 | 62.15 | 90.96 |
| V2A | 34.85 | 24.94 | 20.14 | 84.48 | 82.12 | 56.50 | 89.27 |
| V2B | 29.71 | 24.47 | 18.32 | 83.86 | 86.09 | 64.41 | 92.09 |
| V2C | 31.77 | 24.49 | 19.66 | 84.53 | 84.27 | 59.89 | 93.22 |

### 4.2 相对 V1 的变化（Labeled）
| 版本 | Δ翻译 BLEU | Δ翻译 ROUGE-L | Δ总结 ROUGE-L | Δ总结 BERT-F1 | ΔIFR | ΔStrict |
|---|---:|---:|---:|---:|---:|---:|
| V2A | +3.17 | +0.17 | +0.68 | +0.30 | -2.98 | -5.65 |
| V2B | -1.97 | -0.29 | -1.14 | -0.33 | +0.99 | +2.26 |
| V2C | +0.08 | -0.27 | +0.20 | +0.34 | -0.83 | -2.26 |

解读：
- `V2A` 是“主任务增益换 IF 退化”，不符合本轮目标。
- `V2B` 是“牺牲少量翻译/总结，换取 IF 提升”，方向正确。
- `V2C` 是“主任务最稳，但 labeled IF 略退”。

## 5. 核心指标对比（IF Unlabeled）

### 5.1 绝对分数（Base / V1 / V2A / V2B / V2C）
| 版本 | IFR | Strict | Loose | 可检测约束样本 | 无约束样本 |
|---|---:|---:|---:|---:|---:|
| Base | 50.00 | 40.66 | 53.11 | 273 | 927 |
| V1 (ft) | 65.24 | 57.88 | 68.50 | 273 | 927 |
| V2A | 63.37 | 55.68 | 67.03 | 273 | 927 |
| V2B | 66.04 | 58.24 | 69.96 | 273 | 927 |
| V2C | 66.58 | 59.34 | 69.96 | 273 | 927 |

### 5.2 相对 V1 的变化（IF Unlabeled）
| 版本 | ΔIFR | ΔStrict | ΔLoose |
|---|---:|---:|---:|
| V2A | -1.87 | -2.20 | -1.47 |
| V2B | +0.80 | +0.37 | +1.47 |
| V2C | +1.34 | +1.47 | +1.47 |

解读：
- `V2A` 在无标注 IF 上也退化，进一步确认不推荐。
- `V2B` 与 `V2C` 在无标注 IF 上均优于 V1。
- `V2C` 在 `if_unlabeled` 的 `IFR/Strict` 提升幅度略优于 `V2B`。

## 6. 协议阈值（相对 Base，Labeled）
协议口径（`QWEN3_RIGOROUS_PROTOCOL.md`）：
- 翻译：BLEU +3 或 ROUGE-L +4
- 总结：ROUGE-L +4 且 BERTScore-F1 +1
- IF：IFR +8 且 Strict +5

### 判定结果
| 版本 | 翻译 | 总结 | IF（labeled） | 说明 |
|---|---|---|---|---|
| V1 | 通过 | 通过 | 未通过 | IFR 增益不足 |
| V2A | 通过 | 通过 | 未通过 | IFR/Strict 均弱于 V1 |
| V2B | 通过 | 通过 | 未通过 | IF 三者中最接近，但 IFR 仍不足 |
| V2C | 通过 | 通过 | 未通过 | 主任务稳，IFR 增益不足 |

结论：
- 三个 V2 方案都没有实现“labeled IF 达标闭环”（核心短板仍是 IFR）。
- `V2B` 是本轮最接近 IF 目标的版本。

## 7. 约束维度（重点项）观察

### 7.1 Labeled IF 重点约束（V1 vs V2）
| 约束类型 | V1 | V2A | V2B | V2C |
|---|---:|---:|---:|---:|
| json_format | 0.00 | 100.00 | 50.00 | 0.00 |
| bullet_points | 0.00 | 0.00 | 0.00 | 0.00 |
| placeholder_count | 74.19 | 70.97 | 74.19 | 70.97 |
| exact_paragraphs | 87.50 | 83.33 | 87.50 | 79.17 |
| max_words | 76.92 | 80.77 | 84.62 | 84.62 |
| max_sentences | 80.00 | 80.00 | 100.00 | 80.00 |

### 7.2 IF Unlabeled 重点约束（V1 vs V2）
| 约束类型 | V1 | V2A | V2B | V2C |
|---|---:|---:|---:|---:|
| json_format | 100.00 | 100.00 | 100.00 | 100.00 |
| table_format | 21.05 | 5.26 | 21.05 | 21.05 |
| placeholder_count | 50.00 | 42.86 | 42.86 | 42.86 |
| exact_paragraphs | 75.00 | 62.50 | 87.50 | 56.25 |
| max_words | 81.82 | 81.82 | 90.91 | 81.82 |
| max_sentences | 85.71 | 92.86 | 85.71 | 85.71 |
| markdown_format | 40.00 | 40.00 | 40.00 | 40.00 |

注意：
- 多个约束样本数很小（如 `json_format` / `table_format`），波动较大，不能单独作为版本结论依据。
- 版本选择应以“核心指标 + 双评测一致性”为主。

## 8. 探索性综合 IF 指标（非协议指标，仅辅助决策）
说明：
- IFR 按约束数加权（labeled + unlabeled）
- Strict / Loose 按有约束样本数加权（labeled + unlabeled）

| 版本 | 综合 IFR | 综合 Strict | 综合 Loose |
|---|---:|---:|---:|
| V1 | 77.51 | 59.56 | 77.33 |
| V2A | 74.95 | 56.00 | 75.78 |
| V2B | 78.43 | 60.67 | 78.67 |
| V2C | 77.51 | 59.56 | 79.11 |

解读：
- `V2B` 的综合 IF（IFR/Strict）最佳。
- `V2C` 的综合 Loose 最好，但综合 IFR/Strict 与 V1 基本持平。

## 9. 训练完成情况（核对）
三组均完成训练（`epoch=3.0`, `global_step=675`）：
- `V2A`: `best_step=400`
- `V2B`: `best_step=400`
- `V2C`: `best_step=600`

说明：
- `V2C` 训练总时长明显更短，但 `trainer_state.json` 显示训练完整结束，结果可纳入比较。

## 10. 推荐方案（最终）

### 10.1 推荐方案
- **主候选：V2B（IF 优先）**
  - 原因：在 `labeled + if_unlabeled` 两套 IF 评测上都优于 V1，方向一致。
  - 代价：翻译/总结相对 V1 小幅回退（可接受与否取决于业务优先级）。

### 10.2 备选方案
- **备选：V2C（平衡/保守版）**
  - 原因：翻译/总结更稳，unlabeled IF 也有提升。
  - 缺点：labeled IF 相对 V1 略退，不适合作为“IF 强化版”主线。

### 10.3 不推荐方案
- **V2A**
  - 理由：IF 在两套评测上均退化，和本轮优化目标相反。

## 11. 下一步建议（V3）
1. 固定主线版本为 `V2B`，围绕其继续优化。
2. 数据侧只做“结构/格式约束”专项增强（JSON、列表、占位符、段落结构等），不要再泛化提高 IF 比例。
3. 保留 `V2C` 作为回退版本，避免 IF 优化导致主任务风险扩大。
4. 继续使用 `labeled + if_unlabeled` 双证据链评测，并在报告中保留约束覆盖率说明。

## 12. 一句话结论
- 本轮 V2 实验中，`V2B` 是最值得推进的版本；`V2C` 适合作为稳健备选；`V2A` 应停止投入。
