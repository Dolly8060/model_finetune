# Qwen3 Strict 1-Seed 技术方向报告（Hybrid IF评测口径）
日期：2026-02-25  


---

## 1. 执行摘要

本轮 `qwen3_strict` 1-seed 结果表明：

1. 所有微调方案（`v1a/v1b/v1c/PlanA`）相对 Base 均显著提升，说明严格数据链下的微调路线有效。
2. 混合 adapter 路线中：
   - `v1a` 最稳健（平衡版本）；
   - `v1c` 在当前 Hybrid 口径下 IF 综合表现最好；
   - `v1b` 翻译 BLEU 最好，但综合性略弱于 `v1a/v1c`。
3. 方案A（双 LoRA adapter 路由：TS + IF）已显示明显潜力：
   - 翻译 `ROUGE-L` 最优；
   - 总结接近最优；
   - IF 显著优于 Base，但当前 IF 综合略低于 `v1a/v1c`。
4. 当前建议：
   - 保留 `v1a` 与 `v1c` 作为混合 adapter 主线候选；
   - 继续并行推进 PlanA（双 adapter 路由），重点优化 IF adapter 与路由策略。

---

## 2. 本轮评估报告为什么更有说服力

本报告的 IF 评测覆盖率显著提升，原因是升级了评分器，而不是重新生成模型输出：


1. `IF_unlabeled` 覆盖率提升到 `97.42%`（`1169/1200`）。
2. `IF_labeled` 覆盖率提升到 `61.00%`（`183/300`，Hybrid口径）。

这意味着当前 IF 结论基于更大可评样本集合，稳定性和说服力更强。

---

## 3. 评估对象与版本范围

本报告覆盖以下模型/系统（均为当前 `qwen3_strict` 路线）：

1. `Base`（Qwen3-1.7B 基础模型）
2. `v1a_s2026`（混合 adapter，平衡训练集）
3. `v1b_s2026`（混合 adapter，IF 优先训练集）
4. `v1c_s2026`（混合 adapter，IF 优先训练集 + 更低学习率）
5. `planA_tsif_v1`（双 adapter 路由系统：TS adapter + IF adapter）

说明：

1. 本报告是 `1-seed` 方向验证报告（节约时间、先看大方向）。
2. `v1a/v1b/v1c` 与 `planA_tsif_v1` 的 3-seed 仍可作为下一阶段稳健性验证。

---

## 4. 数据集构成与来源（完整口径）

本节重点回答三个问题：

1. 训练集/验证集/测试集具体由什么组成。
2. 哪些是自生成数据，哪些是公开数据，哪些是内部测试数据。
3. PlanA（双 adapter）是否沿用同一条严格数据链。

### 4.1 Strict 母数据集总体规模（来自 `data/qwen3_strict_manifest.json`）

当前 strict 数据集规模：

| Split | 样本数 |
|---|---:|
| `train_v1a` | 3600 |
| `train_v1b` | 3600 |
| `train_v1c` | 3600 |
| `val` | 360 |
| `test_labeled` | 900 |
| `test_if_unlabeled` | 1200 |

总测试规模（用于汇报口径）：

1. `test_labeled = 900`（有标注，三任务平衡）
2. `test_if_unlabeled = 1200`（prompt-only IF）
3. 测试总计 = `2100`

### 4.2 任务分布（strict）

| Split | Translation | Summarization | IF |
|---|---:|---:|---:|
| `train_v1a` | 1200 | 1200 | 1200 |
| `train_v1b` | 900 | 900 | 1800 |
| `train_v1c` | 900 | 900 | 1800 |
| `val` | 120 | 120 | 120 |
| `test_labeled` | 300 | 300 | 300 |
| `test_if_unlabeled` | 0 | 0 | 1200 |

### 4.3 训练/验证集来源构成（strict 母数据链）

#### 4.3.1 `train_v1a`（平衡版）

| 来源 | 数量 | 性质 |
|---|---:|---|
| `self_generated` | 1241 | 用户自生成（API） |
| `argilla_ifeval` | 882 | 公开 IF 数据 |
| `arxiv` | 521 | 公开总结数据 |
| `opus-100` | 432 | 公开翻译数据 |
| `chinese_generated` | 318 | 本地中文 IF 模板 + API 响应 |
| `wmt19` | 206 | 公开翻译数据 |

#### 4.3.2 `train_v1b` / `train_v1c`（IF优先版）

两者数据构成相同（`v1c` 当前复制自 `v1b` 数据，仅训练超参不同）：

| 来源 | 数量 |
|---|---:|
| `argilla_ifeval` | 1317 |
| `self_generated` | 920 |
| `chinese_generated` | 483 |
| `arxiv` | 394 |
| `opus-100` | 334 |
| `wmt19` | 152 |

说明：

1. `v1b/v1c` 的 IF 样本占比更高（1800/3600）。
2. 存在 IF 重复采样（见 4.7 风险说明）。

#### 4.3.3 `val`（统一验证集）

| 来源 | 数量 |
|---|---:|
| `self_generated` | 130 |
| `argilla_ifeval` | 79 |
| `arxiv` | 44 |
| `chinese_generated` | 41 |
| `opus-100` | 40 |
| `wmt19` | 26 |

### 4.4 测试集来源构成（strict）

#### 4.4.1 `test_labeled`（900，有标注）

任务分层：

1. Translation：300
2. Summarization：300
3. IF：300

来源构成（按样本来源标签）：

| 来源 | 数量 | 说明 |
|---|---:|---|
| `arxiv` | 300 | 总结有标注测试 |
| `opus-100` | 194 | 翻译有标注测试 |
| `wmt19` | 106 | 翻译有标注测试 |
| `lria_follow_zh` | 193 | 内部中文 IF 有标注测试 |
| `lria_follow_en` | 107 | 内部英文 IF 有标注测试 |

#### 4.4.2 `test_if_unlabeled`（1200，prompt-only IF）

来源构成：

| 来源 | 数量 |
|---|---:|
| `ifeval_prompt_only` | 357 |
| `mifeval_en` | 79 |
| `mifeval_zh` | 80 |
| `mifeval_ja` | 74 |
| `mifeval_ar` | 70 |
| `mifeval_de` | 71 |
| `mifeval_fr` | 77 |
| `mifeval_es` | 85 |
| `mifeval_it` | 71 |
| `mifeval_pt` | 77 |
| `mifeval_pl` | 82 |
| `mifeval_ro` | 77 |

### 4.5 数据来源类型（审计口径）

#### 4.5.1 用户确认的自生成数据（API）

1. `data/train.json`
2. `data/val.json`

用途：

1. 作为 strict 训练/验证来源池的一部分（不是固定只做验证）。

#### 4.5.2 本地生成 IF 数据（不是纯公开原始数据）

1. `data/ifeval_full_with_meta.json`

生成方案（工程口径）：

1. 本地生成中文 IF 模板（`chinese_generated`）
2. 通过 API 生成响应
3. 保存为 `ifeval_full_with_meta.json`

用途：

1. strict 训练/验证集 IF 来源之一
2. 不用于测试集

#### 4.5.3 公开下载数据（训练/测试）

公开 IF 训练数据：

1. `argilla/ifeval-like-data` -> `data/argilla_ifeval.json`

公开翻译/总结数据（metadata 保留后再切分）：

1. `ccdv/arxiv-summarization`（`arxiv`）
2. `Helsinki-NLP/opus-100`（`opus-100`）
3. `wmt/wmt19`（`wmt19`）

公开 prompt-only IF 测试数据：

1. `google/IFEval` -> `dataset/IFEval/input_data.jsonl`
2. `PMMEval / m-IFEval` -> `dataset/m-ifeval/PMMEval-mifeval-*.json`

#### 4.5.4 内部测试集来源

1. `dataset/LRIA-Follow_EN/LRIA-Follow_v3_EN.xlsx`
2. `dataset/LRIA-Follow_ZH/LRIA-Follow_v3_ZH.xlsx`

经转换后用于 strict IF 有标注测试：

1. `data/qwen3_strict_internal_if_labeled.json`

### 4.6 PlanA（双 adapter 路由）数据构成

PlanA 不引入新的母数据来源，而是从 strict 母数据集派生。

来源文件（来自 `data/qwen3_strict_dual_adapter_manifest_v1.json`）：

1. `data/qwen3_strict_train.json`
2. `data/qwen3_strict_val.json`

派生规则：

1. `TS adapter` 只使用 `translation + summarization`
2. `IF adapter` 只使用 `instruction_following`

派生后规模：

| 数据集 | 样本数 | 任务 |
|---|---:|---|
| `train_ts` | 2400 | Translation + Summarization |
| `val_ts` | 240 | Translation + Summarization |
| `train_if` | 1200 | IF |
| `val_if` | 120 | IF |

一致性检查结果：

1. `train_partition_complete = true`
2. `val_partition_complete = true`
3. `train_overlap_count = 0`
4. `val_overlap_count = 0`

### 4.7 数据治理风险与边界（当前版必须说明）

1. `v1b/v1c` 存在 IF 重复采样
   - `duplicate_extra_instances = 600`
   - `duplicate_groups = 486`
   - 作用：强化 IF 学习
   - 风险：有效多样性下降，可能加大过拟合/任务权衡风险

2. `IF_labeled` 的 LRIA-Follow 并非纯 IFEval 风格“格式约束”任务
   - 含大量语义/标签型任务
   - 这正是 Hybrid 口径需要 `LRIA fallback judge` 的原因

3. 泄漏检查（strict manifest）
   - `train/val/test_labeled/test_if_unlabeled` 交叉泄漏当前均为 `0`

---

## 5. 评测方法与评分标准（重点）

本节说明“测试集怎么打分、IF 覆盖率怎么来的”。

### 5.1 评测输入与输出

本报告使用的模型输出文件（已生成）位于：

1. `evaluation/output_data/strict_base/*`
2. `evaluation/output_data/strict_3seed/*_s2026_*`
3. `evaluation/output_data/planA_eval/*`

本报告重新评分结果（当前版）位于：

1. `evaluation/performance/strict_hybrid_260225/*/eval_results.json`

### 5.2 翻译/总结评分标准（`test_labeled`）

对 `translation` 与 `summarization` 子集使用：

1. `BLEU`
2. `ROUGE-1/2/L`
3. `BERTScore (P/R/F1)`

说明：

1. 翻译/总结指标依赖 `reference`。
2. 本轮未重跑 `generate.py`，因此这些指标与之前相同（评分器升级主要影响 IF）。

### 5.3 IF 评分标准（核心）

对 `instruction_following` 子集，评分器会先识别“可验证约束”，再计算 IF 指标：

1. `IFR`（Instruction Following Rate）
   - 所有已识别约束中，通过约束的比例
2. `Strict Accuracy`
   - 单样本内所有识别约束全部通过，记为通过
3. `Loose Accuracy`
   - 单样本内通过约束数达到一半及以上，记为通过

### 5.4 IF 覆盖率定义（这部分最重要）

评分器输出字段（见 `eval_results.json` 的 `instruction_following`）：

1. `samples_evaluated`
   - 被识别出至少一个可验证约束的样本数
2. `no_constraint_samples`
   - 未识别出可验证约束的样本数

覆盖率计算方式：

`coverage = samples_evaluated / (samples_evaluated + no_constraint_samples)`

### 5.5 当前报告采用的 IF 评分口径（Hybrid）

本报告采用 Hybrid IF 评分口径，组成如下：

1. 规则型约束识别（regex/规则）
2. `IFEval` 结构化元数据回填（`instruction_id_list + kwargs`）
3. `m-IFEval` 多语种 key 对齐映射至 `IFEval` 元数据
4. `LRIA-Follow` 可选 fallback judge（仅用于提升 `IF_labeled` 覆盖率）

说明：

1. Hybrid 口径不会改变模型输出，只改变 IF 可评样本覆盖与 IF 指标计算集合。
2. `LRIA fallback` 主要影响 `IF_labeled`，对 `IF_unlabeled` 基本不影响。

### 5.6 当前评测覆盖率（本报告口径）

#### `IF_labeled`（300）

1. `samples_evaluated = 183`
2. `no_constraint_samples = 117`
3. 覆盖率 = `61.00%`

#### `IF_unlabeled`（1200）

1. `samples_evaluated = 1169`
2. `no_constraint_samples = 31`
3. 覆盖率 = `97.42%`

解释：

1. `IF_unlabeled` 已远超 `70%` 覆盖率目标。
2. `IF_labeled` 的 61% 覆盖率已经显著提升说服力，但仍有一部分 LRIA 语义型任务难以自动规则验证。

---

## 6. 核心结果总表（当前 Hybrid 口径）

## 6.1 `test_labeled`（900：翻译300 + 总结300 + IF300）

| 模型 | 翻译 BLEU | 翻译 ROUGE-L | 总结 ROUGE-L | 总结 BERT-F1 | IF IFR | IF Strict | IF Loose | IF可评样本 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 2.66 | 9.10 | 10.53 | 76.41 | 12.77 | 12.57 | 13.11 | 183 |
| `v1a_s2026` | 20.21 | 22.36 | 18.87 | 82.67 | 14.89 | 14.21 | 14.75 | 183 |
| `v1b_s2026` | 20.43 | 22.20 | 18.30 | 82.43 | 14.36 | 13.66 | 14.21 | 183 |
| `v1c_s2026` | 19.28 | 22.26 | 19.48 | 82.84 | 14.89 | 13.66 | 14.75 | 183 |
| `planA_tsif_v1`（路由） | 20.38 | 23.02 | 19.07 | 82.79 | 14.36 | 13.66 | 14.21 | 183 |

观察：

1. 所有微调方案相对 Base 显著提升。
2. `v1c` 总结最强。
3. `v1b` 翻译 BLEU 最强。
4. `planA` 翻译 `ROUGE-L` 最强，显示 TS 任务解耦收益。

## 6.2 `test_if_unlabeled`（1200，prompt-only IF）

| 模型 | IFR | Strict | Loose | samples_evaluated | no_constraint_samples |
|---|---:|---:|---:|---:|---:|
| Base | 45.57 | 34.82 | 50.47 | 1169 | 31 |
| `v1a_s2026` | 63.09 | 50.90 | 65.44 | 1169 | 31 |
| `v1b_s2026` | 62.93 | 50.38 | 65.53 | 1169 | 31 |
| `v1c_s2026` | 64.29 | 51.92 | 66.98 | 1169 | 31 |
| `planA_tsif_v1`（路由） | 62.04 | 50.56 | 64.50 | 1169 | 31 |

观察：

1. `v1c` 当前 IF 最强（Hybrid 高覆盖率口径下）。
2. `v1a` 次之且更稳健。
3. `planA` 在 IF 上已明显优于 Base，但仍略低于 `v1a/v1c`。

## 6.3 综合 IF（按可评样本数加权）

加权方式：

1. `labeled IF` 可评样本数 = 183
2. `if_unlabeled IF` 可评样本数 = 1169
3. 按可评样本数对 `IFR / Strict / Loose` 加权

| 模型 | 综合 IFR | 综合 Strict | 综合 Loose |
|---|---:|---:|---:|
| Base | 41.13 | 31.80 | 45.41 |
| `v1a_s2026` | 56.56 | 45.93 | 58.58 |
| `v1b_s2026` | 56.36 | 45.41 | 58.58 |
| `v1c_s2026` | 57.60 | 46.75 | 59.91 |
| `planA_tsif_v1`（路由） | 55.59 | 45.56 | 57.69 |

结论：

1. `v1c` 当前综合 IF 最强。
2. `v1a` 最稳且接近 `v1c`。
3. `planA` 综合 IF 略低于 `v1a/v1c`，但考虑其 TS 侧优势，仍具继续投入价值。

---

## 7. 方案A（双 adapter 路由）解读

### 7.1 方案A当前是什么

`planA_tsif_v1` 不是单一 adapter，而是系统级路由结果：

1. `TS adapter` 负责翻译 + 总结
2. `IF adapter` 负责 IF 任务
3. 在 `test_labeled` 上按任务类型路由后再合并评分

### 7.2 当前优势

1. 翻译 `ROUGE-L` 最佳（23.02）
2. 总结接近最优
3. 证明对 1.7B 模型进行任务拆分训练具有工程价值

### 7.3 当前短板

1. IF 综合仍未超过 `v1c`
2. 部署复杂度更高（双 adapter + 路由）
3. 复合请求（TS + 格式约束）仍需要更细路由策略或 fallback

---

## 8. 当前阶段结论与建议（面向决策）

### 8.1 结论

1. `qwen3_strict` 数据链与微调路线已验证有效。
2. `v1a` 和 `v1c` 是当前最值得保留的混合 adapter 候选。
3. `PlanA` 在 TS 任务上已体现优势，建议继续并行推进。
4. 当前 IF 评测结论的可信度已显著提升（高覆盖率 Hybrid 口径）。

### 8.2 建议的下一步（投入产出比优先）

1. 对外/对上汇报使用本报告（Hybrid口径），同时保留 strict 原口径结果作为技术附录。
2. 混合 adapter 主线优先关注 `v1a` 与 `v1c`。
3. PlanA 继续推进，重点优化 IF adapter 与路由策略。
4. 后续如需更强统计稳健性，再补 `3-seed`。

---

## 9. 关键证据文件（便于复核）

数据构建与构成：

1. `data/qwen3_strict_manifest.json`
2. `data/qwen3_strict_dual_adapter_manifest_v1.json`
3. `README_QWEN3_STRICT_CURRENT.md`

模型输出（已生成）：

1. `evaluation/output_data/strict_base/*`
2. `evaluation/output_data/strict_3seed/*_s2026_*`
3. `evaluation/output_data/planA_eval/*`

本报告使用的重评分结果（Hybrid）：

1. `evaluation/performance/strict_hybrid_260225/*/eval_results.json`

评分脚本（当前口径实现）：

1. `scripts/score.py`

