# 模型评估报告 - BaseQwen3 vs FTQwen3

## 1. 设计目标
该方案强调“可复现 + 可审计 + 可对比”：
- 训练/验证/测试严格隔离
- prompt 级去重，避免数据泄漏
- 标注测试集与无标注 IF 测试集分开评估
- 推理与评分分离（`generate.py` + `score.py`）

## 2. Core Metrics (Base vs FT)
## 翻译子集 (Translation)
| 指标 | BaseQwen3 | FTQwen3 | Delta |
|---|---:|---:|---:|
| BLEU | 3.67 | 31.68 | 28.02 |
| ROUGE-L | 10.32 | 24.77 | 14.45 |
| BERTScore-F1 | 78.20 | 87.87 | 9.67 |

## 总结子集 (Summarization)
| 指标 | BaseQwen3 | FTQwen3 | Delta |
|---|---:|---:|---:|
| BLEU | 2.03 | 9.19 | 7.16 |
| ROUGE-L | 9.34 | 19.46 | 10.12 |
| BERTScore-F1 | 75.80 | 84.18 | 8.39 |

## 指令遵循子集 (Instruction Following)
| 指标 | BaseQwen3 | FTQwen3 | Delta |
|---|---:|---:|---:|
| BLEU | 7.42 | 23.48 | 16.06 |
| ROUGE-L | 14.93 | 26.53 | 11.60 |
| BERTScore-F1 | 84.17 | 88.57 | 4.39 |
| IFR | 81.95 | 85.10 | 3.15 |
| Strict Accuracy | 51.98 | 62.15 | 10.17 |
| Loose Accuracy | 89.27 | 90.96 | 1.69 |

### 统计信息
- samples_evaluated: 177
- no_constraint_samples: 8
- total_constraints: 604
- avg_constraints_per_sample: 3.41
> 注: 8 个样本未检测到可验证的约束（可能是纯问答/总结任务被归入IF类，或约束模式未覆盖）。
> IFR/Strict Acc/Loose Acc 仅基于 177 个有约束的样本计算。

## 8. 推荐判定阈值（可按业务调整）
- 翻译：BLEU 提升 >= 3 或 ROUGE-L 提升 >= 4
- 总结：ROUGE-L 提升 >= 4 且 BERTScore-F1 提升 >= 1
- 指令遵循：IFR 提升 >= 8 且 Strict 提升 >= 5

- Translation check: PASS (BLEU +28.02, ROUGE-L +14.45)
- Summarization check: PASS (ROUGE-L +10.12, BERTScore-F1 +8.39)
- Instruction Following check: PARTIAL (IFR +3.15, Strict +10.17)

## 9. Constraint Delta (Instruction Following)
Top improvements:
- keyword_exclude: 0.00 -> 100.00 (+100.00)
- zh_no_degree_adverbs: 0.00 -> 100.00 (+100.00)
- exact_paragraphs: 20.83 -> 87.50 (+66.67)
- max_words: 11.54 -> 76.92 (+65.38)
- max_sentences: 20.00 -> 80.00 (+60.00)
Largest regressions:
- bullet_points: 75.00 -> 0.00 (-75.00)
- json_format: 50.00 -> 0.00 (-50.00)
- placeholder_count: 100.00 -> 74.19 (-25.81)
- end_with_question: 100.00 -> 75.00 (-25.00)
- min_words: 100.00 -> 78.57 (-21.43)

## 10. Conclusion
- Base vs FT evidence supports effective finetuning, especially on translation and summarization.
- Instruction Following shows clear Strict gains, but IFR gain is below the recommended +8 threshold.
- For stronger rigor, run 3 seeds and report mean/std as defined in protocol section 7.
