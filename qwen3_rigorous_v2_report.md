# Qwen3-1.7B 三版本评估报告（初版，面向汇报）

> 版本范围：Qwen3 基础模型（Base）/ 微调模型 V1（FT）/ 微调模型 V2B（IF 优先版）  
> 报告目的：用于向技术同事与管理层汇报阶段性成果，同时明确数据链、证据链与风险边界。

## 1. 结论先行（管理层摘要）

1. 本轮 LoRA 微调路线已被验证有效：`V1` 相对 `Base` 在翻译、总结、指令遵循三个维度均有显著提升。
2. `V2B` 在 `V1` 基础上继续提升指令遵循（IF），且提升在 `labeled` 与 `if_unlabeled` 两套评测上方向一致。
3. `V2B` 的代价是翻译与总结指标相对 `V1` 出现小幅回退，属于有意识的多任务配比 trade-off。
4. 当前核心瓶颈已从“是否有效提升”转移到“如何进一步提升格式/结构类约束稳定性（IFR）且不伤主任务”。

一句话建议：
- 若业务优先级是 IF 强化，建议以 `V2B` 为主线继续优化；
- 若业务优先级是翻译/总结稳定性，建议保留 `V1` 作为生产基线，`V2B` 作为下一轮候选。

## 2. 先回应你的方法论担忧：为什么“只用自己文件证明自己”不够

你的判断是对的。单纯引用本工程内部产物（评估结果、日志、配置）只能证明“在本工程流程内的一致性”，不能自动证明“对外部世界的真实性与可复现性”。

因此本报告采用分层证据链，而不是只做内部自证：

1. **外部来源层（公开数据与公开基准）**
- 协议中明确了推荐的公开数据/基准来源：OPUS-100、WMT19、arXiv Summarization、Argilla IFEval-like、HuggingFaceH4/IFEval（`QWEN3_RIGOROUS_PROTOCOL.md:19`）。
- 这层回答“数据从哪里来、理论上应该如何构建严谨评测”。

2. **项目数据治理层（脚本与manifest）**
- 用 `scripts/build_qwen3_rigorous_dataset.py` 定义数据加载、去重、隔离、抽样、泄漏检查流程（`scripts/build_qwen3_rigorous_dataset.py:108`, `scripts/build_qwen3_rigorous_dataset.py:183`, `scripts/build_qwen3_rigorous_dataset.py:330`, `scripts/build_qwen3_rigorous_dataset.py:451`）。
- 用 `data/qwen3_rigorous_manifest.json` 固化本次实际构建结果（规模、来源构成、池统计、泄漏检查），这是“审计工件”。

3. **实验执行层（配置、训练状态、评测产物）**
- 配置文件定义模型、超参和训练集版本（`configs/finetune_qwen3_lora_rigorous.yaml`, `configs/finetune_qwen3_lora_v2b.yaml`）。
- 训练状态文件证明训练确实完成、best checkpoint 与 loss 轨迹（`outputs/.../trainer_state.json`）。
- 评估结果文件给出最终指标（`evaluation/rigorous/*/eval_results.json`）。

4. **结论层（本报告）**
- 明确哪些结论是“内部可复核但尚需外部复验”，哪些结论已经具备较高可信度。

这意味着：本报告不会把“内部结果”直接当成“外部真理”，而是给出一条可审计、可挑战、可扩展的论证链。

## 3. 报告范围与版本映射

用户口径与目录口径映射如下：

1. `qwen3基础模型` -> `base_labeled` + `base_if_unlabeled`
2. `qwen3微调模型v1` -> `ft_labeled` + `ft_if_unlabeled`
3. `qwen3微调模型v2b` -> `v2b_labeled` + `v2b_if_unlabeled`

对应核心结果文件：
- `evaluation/rigorous/base_labeled/eval_results.json`
- `evaluation/rigorous/base_if_unlabeled/eval_results.json`
- `evaluation/rigorous/ft_labeled/eval_results.json`
- `evaluation/rigorous/ft_if_unlabeled/eval_results.json`
- `evaluation/rigorous/v2b_labeled/eval_results.json`
- `evaluation/rigorous/v2b_if_unlabeled/eval_results.json`

## 4. 数据链（从上游来源到最终评测集）的完整说明

### 4.1 上游来源层（公开来源与项目内快照）

协议定义的公开来源与角色（方法论层）：
- 翻译：OPUS-100 / WMT19（`QWEN3_RIGOROUS_PROTOCOL.md:19`）
- 总结：arXiv Summarization（`QWEN3_RIGOROUS_PROTOCOL.md:19`）
- IF训练：Argilla IFEval-like（`QWEN3_RIGOROUS_PROTOCOL.md:23`）
- IF评测（prompt-only）：HuggingFaceH4/IFEval（`QWEN3_RIGOROUS_PROTOCOL.md:24`）

项目实际使用的是这些来源的本地化快照/二次整理文件（构建脚本可见）：
- 训练候选源：`train.json`, `val.json`, `train_v3.json`, `val_v3.json`, `train_mixed_3k.json`, `argilla_ifeval.json`, `ifeval_full_with_meta.json`（`scripts/build_qwen3_rigorous_dataset.py:254-262`）
- 标注评测候选源：`public_val_v2.json`, `test_v4_enhanced.json`（`scripts/build_qwen3_rigorous_dataset.py:267-270`）
- 无标注 IF 候选源：`dataset/m-ifeval/PMMEval-mifeval-*.json` + `dataset/IFEval/input_data.jsonl`（`scripts/build_qwen3_rigorous_dataset.py:124-180`, `scripts/build_qwen3_rigorous_dataset.py:305-307`）

说明：
- 这一步仍然依赖项目内快照文件，因此若要提升“对外证明力”，后续应补充原始下载脚本、版本号和文件哈希（见本报告第 12 节建议）。

### 4.2 数据治理层（去重、隔离、采样、泄漏检查）

构建脚本的关键机制：

1. **统一样本键（prompt级去重）**
- `dedup_key = instruction[:240] + input[:120]`（`scripts/build_qwen3_rigorous_dataset.py:66-68`）
- 这是本项目“prompt级去重”的具体实现，对应协议中的“prompt-level deduplication”（`QWEN3_RIGOROUS_PROTOCOL.md:14`）

2. **任务识别与分流**
- 若没有 `task_type`，脚本会按关键词/约束模式推断 translation/summarization/instruction_following（`scripts/build_qwen3_rigorous_dataset.py:79-92`）

3. **先建测试集，再过滤训练集重叠**
- 脚本先构建 `labeled test` 和 `if_unlabeled test`（`scripts/build_qwen3_rigorous_dataset.py:273-330`）
- 再用 `test_keys` 过滤训练候选，累计 `skipped_test_overlap`（`scripts/build_qwen3_rigorous_dataset.py:330`, `scripts/build_qwen3_rigorous_dataset.py:349-351`）

4. **Train/Val/Test 全局泄漏检查**
- 最终对 train/val/test_labeled/test_if_unlabeled 做键交集检查（`scripts/build_qwen3_rigorous_dataset.py:445-458`）
- 写入 `manifest` 的 `leakage` 字段（`scripts/build_qwen3_rigorous_dataset.py:451-495`）

### 4.3 本次严谨切分的实际产物（manifest审计结果）

`data/qwen3_rigorous_manifest.json` 记录了本次切分的实际结果：

1. 样本规模（`data/qwen3_rigorous_manifest.json:14-19`）
- train = 3600
- val = 360
- test_labeled = 585
- test_if_unlabeled = 1200

2. 任务分布（`data/qwen3_rigorous_manifest.json:20-39`）
- train：翻译/总结/IF 各 1200（均衡）
- val：各 120（均衡）
- test_labeled：翻译 200 / 总结 200 / IF 185
- test_if_unlabeled：IF 1200

3. 来源分布（`data/qwen3_rigorous_manifest.json:40-73`）
- train/val/test_labeled/test_if_unlabeled 的来源构成都有审计记录

4. 池统计与剔除（`data/qwen3_rigorous_manifest.json:74-86`）
- `skipped_test_overlap = 685`
- 说明构建时确实剔除了与测试重叠样本

5. 泄漏检查（`data/qwen3_rigorous_manifest.json:87-94`）
- train/val/test_labeled/test_if_unlabeled 两两交叉均为 0

## 5. 评估设计与口径（避免“指标好看但不公平”）

### 5.1 双测试链设计（Labeled + IF Unlabeled）

协议采用双证据链评估：

1. `labeled` 严格集：同时评估翻译、总结、IF（有标准答案）
2. `if_unlabeled` prompt-only IF 集：专门评估约束遵循与泛化（无标准答案）

这避免了只看单一测试集带来的偏差，见 `QWEN3_RIGOROUS_PROTOCOL.md:56-103`。

### 5.2 评分脚本口径（IFR / Strict / Loose / 覆盖率）

`score.py` 对 IF 的口径是：
- `instruction_following_rate`（IFR）
- `strict_accuracy`
- `loose_accuracy`
- `no_constraint_samples`
- `by_constraint_type`

见 `scripts/score.py:499-506`。

并且报告中会显式计算约束覆盖率（`coverage`），说明 IF 指标只在“可检测约束子集”上计算，见 `scripts/score.py:892`, `scripts/score.py:901`。

## 6. 训练集来源构成（你特别关心的部分）

> 这一节是对上版报告的补充重点：不仅给总量，还给任务-来源交叉构成，并单独分析 V2B 的风险。

### 6.1 V1（`qwen3_rigorous_train.json`）训练集来源构成

总样本数：3600（无重复样本）

#### 任务构成
- IF: 1200（33.33%）
- 总结: 1200（33.33%）
- 翻译: 1200（33.33%）

#### 来源构成（总体）
- `self_generated`: 1281（35.58%）
- `v3_dataset`: 973（27.03%）
- `argilla_ifeval`: 872（24.22%）
- `chinese_generated`: 328（9.11%）
- `mixed_3k`: 146（4.06%）

#### 任务-来源交叉（V1）
- IF（1200）
  - `argilla_ifeval`: 872（72.67%）
  - `chinese_generated`: 328（27.33%）
- 总结（1200）
  - `self_generated`: 730（60.83%）
  - `v3_dataset`: 350（29.17%）
  - `mixed_3k`: 120（10.00%）
- 翻译（1200）
  - `v3_dataset`: 623（51.92%）
  - `self_generated`: 551（45.92%）
  - `mixed_3k`: 26（2.17%）

### 6.2 V2B（`qwen3_rigorous_train_v2b.json`）训练集来源构成

总样本数：3600（**存在重复采样**，详见风险节）

#### 任务构成（V2B 的核心变化）
- IF: 1800（50.00%）
- 总结: 900（25.00%）
- 翻译: 900（25.00%）

这与 V2 计划文档中 `V2B = IF 50% / 翻译 25% / 总结 25%` 一致（`evaluation/rigorous/V2_PLAN.md:23`, `evaluation/rigorous/V2_PLAN.md:53`）。

#### 来源构成（总体）
- `argilla_ifeval`: 1310（36.39%）
- `self_generated`: 970（26.94%）
- `v3_dataset`: 721（20.03%）
- `chinese_generated`: 490（13.61%）
- `mixed_3k`: 109（3.03%）

#### 任务-来源交叉（V2B）
- IF（1800）
  - `argilla_ifeval`: 1310（72.78%）
  - `chinese_generated`: 490（27.22%）
- 总结（900）
  - `self_generated`: 550（61.11%）
  - `v3_dataset`: 259（28.78%）
  - `mixed_3k`: 91（10.11%）
- 翻译（900）
  - `v3_dataset`: 462（51.33%）
  - `self_generated`: 420（46.67%）
  - `mixed_3k`: 18（2.00%）

### 6.3 标注测试集与 IF-unlabeled 测试集来源构成（用于解释评估可信度）

#### `test_labeled`（585）
- `public_v2_eval`: 297
- `test_v4_enhanced`: 288

见 `data/qwen3_rigorous_manifest.json:55-58`。

#### `test_if_unlabeled`（1200）
- `ifeval_prompt_only`: 354
- 其余来自 PMMEval 多语种 m-IFEval（ar/fr/pl/it/de/zh/pt/ja/ro/es/en 等）

见 `data/qwen3_rigorous_manifest.json:59-72`。

这意味着 IF 泛化评估并非只看英文单一来源，而是包含多语种 prompt-only 场景（尽管后续仍需提高约束检测覆盖率）。

## 7. V2B 训练集是否有风险？答案：有，而且是“可控但必须正视”的风险

你问得很关键。`V2B` 的训练集确实存在风险，主要来自“IF优先重配 + 重复采样”的设计。下面给出明确判断。

### 7.1 风险判断（结论）

**结论：V2B 有风险，但目前表现为“方向正确、代价可见、可继续优化”的工程风险，不是立即否定版本的致命风险。**

### 7.2 主要风险点（按重要性排序）

#### 风险 A：重复采样导致训练集有效多样性下降（高优先级）

实际统计结果（对 `data/qwen3_rigorous_train_v2b.json` 的去重检查）：
- 总样本 3600
- 唯一样本约 3000
- 额外重复实例约 600（占 16.67%）

含义：
- `V2B` 为了把 IF 提升到 1800 条，采用了重采样（计划文档中的 `random.sample / random.choices` 思路），会提高 IF 样式的曝光频次，但也会增加过拟合和“格式模板记忆”的风险。
- 这与 `V2B` 在 IF 上提升、翻译总结小幅回退的现象是一致的（符合因果逻辑）。

#### 风险 B：任务权重变化导致主任务回退（高优先级）

`V2B` 将 IF 比例从 33.33% 提升到 50%，翻译/总结从各 33.33% 降到 25%。

实际结果也出现对应 trade-off：
- 相比 `V1`，`V2B` 的翻译/总结略降（见第 10 节结果对比）
- 但 IF 在两套评测上均提升

这不是“坏结果”，而是“目标函数改变后符合预期的代价”。问题在于是否符合当前业务优先级。

#### 风险 C：IF 训练来源偏向少数来源（中优先级）

`V2B` 的 IF 样本中：
- `argilla_ifeval` ~72.78%
- `chinese_generated` ~27.22%

风险点：
- IF 强化很可能更多学习了这两类数据的表达风格与约束模式；
- 若线上真实指令格式分布与这两类来源差异较大，可能产生泛化偏差。

缓解信号：
- `if_unlabeled` 使用了 IFEval + PMMEval 多语种 prompt-only 测试，能部分验证跨来源泛化方向（不是完全消除风险）。

#### 风险 D：IF-unlabeled 约束覆盖率较低，结论代表性有限（中优先级）

三版本在 `if_unlabeled` 上一致出现：
- 总样本 1200
- 可检测约束样本 273
- 覆盖率仅 22.75%

含义：
- IFR/Strict/Loose 的提升真实存在，但严格来说仅能代表“可检测约束子集”的泛化趋势；
- 不能直接外推为对全部 1200 条 prompt 的完整 IF 能力结论。

#### 风险 E：内部证据链强、外部复验链不足（方法论风险）

当前证据链在工程内部是完整的，但对外部审计仍缺：
- 原始公开数据下载脚本与版本固定（commit/hash）
- 原始文件哈希与快照登记
- 第三方复跑结果
- 多 seed 统计显著性

这会影响“对外证明力”，但不影响你在内部推进技术路线与下一轮实验。

## 8. 实验设置与公平性（V1 vs V2B 的因果归因基础）

### 8.1 配置对比：基本控制变量成立

`V1` 与 `V2B` 训练超参基本一致，核心差异主要是训练集版本与输出目录：
- `dataset`: `qwen3_rigorous_train` vs `qwen3_rigorous_train_v2b`
- `eval_dataset`: 均为 `qwen3_rigorous_val`
- LoRA、batch、LR、epoch 相同

证据：
- `configs/finetune_qwen3_lora_rigorous.yaml:16-17`, `configs/finetune_qwen3_lora_rigorous.yaml:26-28`, `configs/finetune_qwen3_lora_rigorous.yaml:38-43`
- `configs/finetune_qwen3_lora_v2b.yaml:15-16`, `configs/finetune_qwen3_lora_v2b.yaml:25-27`, `configs/finetune_qwen3_lora_v2b.yaml:37-42`

因此：`V1 -> V2B` 的性能变化主要可归因于数据配比变化（而非超参漂移）。

### 8.2 数据集注册（工具链层面）

`V2B` 训练集在 `data/dataset_info.json` 中正式注册，而非临时替换：
- `qwen3_rigorous_train`：`data/dataset_info.json:137`
- `qwen3_rigorous_val`：`data/dataset_info.json:146`
- `qwen3_rigorous_train_v2b`：`data/dataset_info.json:182`

## 9. 训练过程证据（V1 vs V2B）

### 9.1 V1（`outputs/qwen3-1.7B-lora-rigorous`）

- `global_step = 675`，`epoch = 3.0`（完成训练）
- `best_global_step = 600`
- `best_metric(eval_loss) = 1.109719`
- `train_loss = 1.111405`
- `train_runtime = 2639.49s`（约 43m59s）

证据：`outputs/qwen3-1.7B-lora-rigorous/trainer_state.json:2-7`, `outputs/qwen3-1.7B-lora-rigorous/trainer_state.json:450`, `outputs/qwen3-1.7B-lora-rigorous/trainer_state.json:509-512`。

### 9.2 V2B（`outputs/qwen3-1.7B-lora-v2b`）

- `global_step = 675`，`epoch = 3.0`（完成训练）
- `best_global_step = 400`
- `best_metric(eval_loss) = 1.126866`
- `step=600` 时 `eval_loss` 回升到 `1.129115`
- `train_loss = 1.043096`
- `train_runtime = 4964.91s`（约 1h22m45s）

证据：`outputs/qwen3-1.7B-lora-v2b/trainer_state.json:2-7`, `outputs/qwen3-1.7B-lora-v2b/trainer_state.json:302`, `outputs/qwen3-1.7B-lora-v2b/trainer_state.json:450`, `outputs/qwen3-1.7B-lora-v2b/trainer_state.json:509-512`。

### 9.3 训练过程解读（为什么 loss 与业务指标不一致）

观察到：
- `V2B` 的 `train_loss` 更低，但在平衡验证集上的 `eval_loss` 不如 `V1`
- 同时业务 IF 指标（尤其 labeled/unlabeled 双评测）优于 `V1`

这说明：
- `V2B` 更像“定向优化版本”，其业务目标与平衡验证集平均 loss 并不完全一致；
- 不能只用 `eval_loss` 排序版本，必须看多任务业务指标与双证据链一致性。

## 10. 核心评估结果（Base / V1 / V2B）

### 10.1 Labeled 严格集（翻译+总结+IF）

| 版本 | 翻译 BLEU | 翻译 ROUGE-L | 总结 ROUGE-L | 总结 BERT-F1 | IF IFR | IF Strict | IF Loose |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 3.67 | 10.32 | 9.34 | 75.80 | 81.95 | 51.98 | 89.27 |
| V1 | 31.68 | 24.77 | 19.46 | 84.18 | 85.10 | 62.15 | 90.96 |
| V2B | 29.71 | 24.47 | 18.32 | 83.86 | 86.09 | 64.41 | 92.09 |

关键结果文件位置：
- `evaluation/rigorous/base_labeled/eval_results.json:2`
- `evaluation/rigorous/ft_labeled/eval_results.json:2`
- `evaluation/rigorous/v2b_labeled/eval_results.json:2`

### 10.2 IF Unlabeled（prompt-only，IF 泛化）

| 版本 | IFR | Strict | Loose | 可检测约束样本 | 无可检测约束样本 |
|---|---:|---:|---:|---:|---:|
| Base | 50.00 | 40.66 | 53.11 | 273 | 927 |
| V1 | 65.24 | 57.88 | 68.50 | 273 | 927 |
| V2B | 66.04 | 58.24 | 69.96 | 273 | 927 |

关键结果文件位置：
- `evaluation/rigorous/base_if_unlabeled/eval_results.json:10-15`
- `evaluation/rigorous/ft_if_unlabeled/eval_results.json:10-15`
- `evaluation/rigorous/v2b_if_unlabeled/eval_results.json:10-15`

### 10.3 V2B 相对 V1 的业务增量（最重要）

#### Labeled
- 翻译 BLEU：`-1.97`
- 翻译 ROUGE-L：`-0.29`
- 总结 ROUGE-L：`-1.14`
- 总结 BERT-F1：`-0.33`
- IF IFR：`+0.99`
- IF Strict：`+2.26`
- IF Loose：`+1.13`

#### IF Unlabeled
- IFR：`+0.80`
- Strict：`+0.37`
- Loose：`+1.47`

解释：
- 这正是一个“IF 优先版”应该呈现的形态：主任务轻微回退，IF 稳定提升，且在两套 IF 证据链上方向一致。

## 11. 协议阈值判定（帮助向领导解释“进展到哪一步”）

协议建议阈值（相对 Base）：
- 翻译：BLEU +3 或 ROUGE-L +4
- 总结：ROUGE-L +4 且 BERT-F1 +1
- IF（labeled）：IFR +8 且 Strict +5

见 `QWEN3_RIGOROUS_PROTOCOL.md:120-123`。

### 11.1 V1 判定（相对 Base）
- 翻译：通过
- 总结：通过
- IF（labeled）：**未通过**（Strict 达标，但 IFR 增益不足 +8）

### 11.2 V2B 判定（相对 Base）
- 翻译：通过
- 总结：通过
- IF（labeled）：**未通过**（Strict 达标，但 IFR 增益仍不足 +8）

这说明：
- “微调有效性”已经明确成立；
- 当前未闭环的是 IF 的高标准验收（尤其 IFR），下一轮应聚焦格式/结构约束专项增强，而不是继续泛化调高 IF 比例。

## 12. 为什么 V2B 还值得继续推进（而不是因为有风险就放弃）

即使存在上述风险，`V2B` 仍有推进价值，理由是：

1. 风险与收益高度一致（可解释）
- 由于 IF 比例提高与重复采样，出现 IF 提升 + 主任务小幅回退，这是可解释的，不是随机波动式异常。

2. 提升不是单一评测偶然值
- `labeled IF` 与 `if_unlabeled IF` 两套评测都向同一方向提升。

3. 风险可工程化缓解
- 重复采样可替换为“去重后模板扩写”或“loss reweighting”；
- IF 短板约束类型已可定位，下一轮可做专项数据增强。

## 13. 下一步建议（优先级排序，面向下一轮实验）

### 13.1 数据与训练层（优先级最高）

1. **不要再泛化提高 IF 比例**，改做“格式/结构约束专项增强”
- 重点覆盖：`quotation_wrap`, `table_format`, `markdown_format`, `placeholder_count`, `bullet_points`, `start_with`, `no_commas`, `all_uppercase`
- 原因：这些约束是 IFR 的主要扣分项，也是业务可感知风险点

2. **降低 V2B 的重复采样依赖**
- 方案 A：去重后扩写高价值 IF 样本（模板变体）
- 方案 B：保持样本不重复，改为 loss 权重/采样权重倾斜
- 目标：保留 IF 提升方向，降低对主任务的伤害

3. **增加更贴近业务的 IF 验证集**
- 单独构建“格式/结构约束验证集”（JSON、表格、占位符、列表等）
- 避免只看综合 IF 指标掩盖关键短板

### 13.2 评估与可信度层（面向“非自证”）

4. **补齐外部复核链（建议作为对外汇报升级版）**
- 固定原始公开数据下载脚本与版本号
- 输出每份原始数据文件 SHA256
- 输出构建后 split 的 SHA256
- 形成“数据血缘表（source -> local snapshot -> split -> eval）”

5. **做 3-seed 实验并报告均值/方差**
- 协议已建议 `2026/2027/2028`（`QWEN3_RIGOROUS_PROTOCOL.md:127`）
- 这一步能显著提升对技术同事和领导的说服力

6. **引入一个外部独立holdout（不参与任何本地构建）**
- 用于验证“项目内调参是否过度贴合本地评测集”
- 这是缓解“自己证明自己”最直接的方法

## 14. 汇报建议话术（可直接用）

### 14.1 面向技术同事
- 我们没有只看单一分数，而是用了 `labeled + prompt-only IF` 双证据链。
- 数据构建有脚本化去重和泄漏检查，manifest 显示 0 泄漏。
- V2B 的 IF 提升与数据配比变化一致，但存在 16.67% 重复采样带来的风险，下一轮会改成更稳的专项增强方案。

### 14.2 面向领导
- 本轮已经证明微调路线有效（Base -> V1 大幅提升）。
- V2B 是在不大幅牺牲主任务的前提下进一步提升指令遵循的版本。
- 下一阶段重点是“把格式约束稳定性做扎实”，并增加外部复核手段，让成果更容易对外说明。

## 15. 附录：关键证据文件清单（便于复核）

### 协议与方法
- `QWEN3_RIGOROUS_PROTOCOL.md`
- `scripts/build_qwen3_rigorous_dataset.py`
- `scripts/score.py`

### 数据审计工件
- `data/qwen3_rigorous_manifest.json`
- `data/qwen3_rigorous_train.json`
- `data/qwen3_rigorous_train_v2b.json`
- `data/qwen3_rigorous_val.json`
- `data/qwen3_rigorous_test_labeled.json`
- `data/qwen3_rigorous_test_if_unlabeled.json`

### 训练配置与训练状态
- `configs/finetune_qwen3_lora_rigorous.yaml`
- `configs/finetune_qwen3_lora_v2b.yaml`
- `outputs/qwen3-1.7B-lora-rigorous/trainer_state.json`
- `outputs/qwen3-1.7B-lora-v2b/trainer_state.json`

### 评估结果
- `evaluation/rigorous/base_labeled/eval_results.json`
- `evaluation/rigorous/base_if_unlabeled/eval_results.json`
- `evaluation/rigorous/ft_labeled/eval_results.json`
- `evaluation/rigorous/ft_if_unlabeled/eval_results.json`
- `evaluation/rigorous/v2b_labeled/eval_results.json`
- `evaluation/rigorous/v2b_if_unlabeled/eval_results.json`

---

## 16. 本报告的边界（明确声明）

这份“初版”报告已经能支持内部技术汇报与阶段成果汇报，但若要作为对外正式材料（或更高强度审查场景），仍建议补齐：

1. 外部原始数据快照哈希与下载脚本
2. 多 seed 统计
3. 外部独立 holdout 复评
4. 第三方复跑记录（至少一轮）

这样可以把当前“内部可复核的强证据链”升级为“对外可审计的强证据链”。
