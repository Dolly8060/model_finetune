# Qwen3 Rigorous Evaluation Report (BaseQwen3 vs FTQwen3)

## 1. Scope
This report compares BaseQwen3 and FTQwen3 on the same rigorous labeled test split using the same scoring pipeline (`scripts/score.py`).

## 2. Core Metrics (Base vs FT)
### Translation
| Metric | Base | FT | Delta |
|---|---:|---:|---:|
| BLEU | 3.67 | 31.68 | 28.02 |
| ROUGE-L | 10.32 | 24.77 | 14.45 |
| BERTScore-F1 | 78.20 | 87.87 | 9.67 |

### Summarization
| Metric | Base | FT | Delta |
|---|---:|---:|---:|
| BLEU | 2.03 | 9.19 | 7.16 |
| ROUGE-L | 9.34 | 19.46 | 10.12 |
| BERTScore-F1 | 75.80 | 84.18 | 8.39 |

### Instruction Following
| Metric | Base | FT | Delta |
|---|---:|---:|---:|
| BLEU | 7.42 | 23.48 | 16.06 |
| ROUGE-L | 14.93 | 26.53 | 11.60 |
| BERTScore-F1 | 84.17 | 88.57 | 4.39 |
| IFR | 81.95 | 85.10 | 3.15 |
| Strict Accuracy | 51.98 | 62.15 | 10.17 |
| Loose Accuracy | 89.27 | 90.96 | 1.69 |

## 3. Acceptance Check (Protocol)
- Translation target (BLEU +3.0 or ROUGE-L +4.0): PASS
- Summarization target (ROUGE-L +4.0 and BERTScore-F1 +1.0): PASS
- Instruction Following target (IFR +8.0 and Strict +5.0): PARTIAL / NOT FULLY MET

## 4. IF Constraint Delta
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

## 5. Conclusion
- The finetuning result is clearly positive overall, especially for translation and summarization.
- Instruction-following quality improved, but IFR gain remains below the recommended threshold.
- To strengthen defensibility, run multi-seed experiments and report mean/std.
