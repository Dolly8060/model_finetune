# Qwen3 Strict 当前版本说明（审计/复现实用版）

本文件描述当前 `qwen3_strict` 数据集构建与实验脚手架的实际状态，重点覆盖：

1. 严格集的数据来源（直接输入 vs 上游来源）
2. 原始数据保存目录与关键文件
3. 构建/训练/评测执行命令（`conda` 环境）
4. 当前产物清单与统计结果
5. 已知风险与注意事项

适用场景：
- 技术同事复现
- 审计/合规说明
- 后续训练与评测执行前确认口径

---

## 1. 当前状态摘要（截至当前版本）

当前已完成：

1. 新 `strict` 系列数据集重建（审计模式）
2. 新命名版本映射（`v1a/v1b/v1c`）
3. 内部 IF 测试集（`LRIA-Follow`）转换
4. 公开翻译/总结数据下载并切分为 strict 专用源文件
5. 3-seed 训练/评测/聚合脚手架

当前 strict 数据规模（实际构建结果）：

- `train_v1a`: 3600
- `train_v1b`: 3600
- `train_v1c`: 3600
- `val`: 360
- `test_labeled`: 900
- `test_if_unlabeled`: 1200

说明：
- 总测试规模是 `2100`（`900` 有标注 + `1200` prompt-only IF）
- `v1a` 平衡；`v1b/v1c` 为 IF 优先

---

## 2. 版本命名映射（新系列）

`strict` 系列不再沿用旧命名：

- `v1a` = 旧 `v1` 方向（平衡训练集）
- `v1b` = 旧 `v2b` 方向（IF 优先训练集）
- `v1c` = 旧 `v2c` 方向（IF 优先训练集 + 更低 LR 配置）

注意：
- 当前 `v1c` 数据文件默认复制自 `v1b`（数据相同）
- `v1b`/`v1c` 的差异主要在训练配置（尤其学习率）

---

## 3. 目录与文件职责（原始数据 / 构建产物 / 脚本）

### 3.1 原始数据与中间数据目录

#### `data/`（构建输入与构建输出）
包含：

- 用户自生成训练源：
  - `data/train.json`
  - `data/val.json`
- 公开/生成 IF 训练源：
  - `data/argilla_ifeval.json`
  - `data/ifeval_full_with_meta.json`
- 公开翻译/总结元信息数据（strict 切分前后）：
  - `data/public_eval_with_meta.json`
  - `data/public_train_strict_with_meta.json`
  - `data/public_val_strict_with_meta.json`
  - `data/public_test_strict_with_meta.json`
- strict 构建输出：
  - `data/qwen3_strict_*.json`
  - `data/qwen3_strict_manifest.json`
- 审计工件：
  - `data/audit/qwen3_strict_*`

#### `dataset/`（内部测试源与公开 prompt-only 测试源）
包含：

- 内部 IF 有标注原始测试源：
  - `dataset/LRIA-Follow_EN/LRIA-Follow_v3_EN.xlsx`
  - `dataset/LRIA-Follow_ZH/LRIA-Follow_v3_ZH.xlsx`
- 公开 IF prompt-only 测试源：
  - `dataset/IFEval/input_data.jsonl`
  - `dataset/m-ifeval/PMMEval-mifeval-*.json`
- 内部翻译回放/对照（当前不作为主有标注测试）：
  - `dataset/LRIATrans/*.json`

### 3.2 配置目录

#### `configs/`
- `configs/finetune_qwen3_lora_strict_v1a.yaml`
- `configs/finetune_qwen3_lora_strict_v1b.yaml`
- `configs/finetune_qwen3_lora_strict_v1c.yaml`

#### `configs/generated_strict_3seed/`
- 由 3-seed 训练脚本自动生成（seed 化后的 yaml）

### 3.3 脚本目录（strict 相关）

- `scripts/convert_lria_follow_to_strict_test.py`
- `scripts/prepare_qwen3_strict_public_sources.py`
- `scripts/build_qwen3_strict_dataset.py`
- `scripts/run_qwen3_strict_3seed_train.ps1`
- `scripts/run_qwen3_strict_3seed_eval.ps1`
- `scripts/aggregate_qwen3_strict_3seed.py`

---

## 4. 严格集的数据来源说明（重点）

这里严格区分两个概念：

1. **直接输入（Direct Inputs）**
2. **上游来源（Upstream Lineage）**

### 4.1 当前 strict 构建的直接输入文件（`build_qwen3_strict_dataset.py` 实际读取）

当前构建 (`data/qwen3_strict_manifest.json`) 的 direct inputs 为：

- `data/train.json`
- `data/val.json`
- `data/argilla_ifeval.json`
- `data/public_train_strict_with_meta.json`
- `data/ifeval_full_with_meta.json`
- `data/qwen3_strict_internal_if_labeled.json`
- `data/public_test_strict_with_meta.json`
- `dataset/IFEval/input_data.jsonl`
- `dataset/m-ifeval/PMMEval-mifeval-*.json`

说明：
- `public_val_strict_with_meta.json` 当前未纳入训练（默认配置 `--include-public-val-in-train` 为 false）
- `dataset/LRIATrans/*.json` 当前不作为 strict 主测试源

### 4.2 上游来源与来源类型（用于审计说明）

### 4.2.1 数据来源台账（按“自生成 / 公开下载 / 内部测试源”明确标注）

下表用于审计/汇报场景，明确每类数据是否为用户自生成，以及具体生成/下载方案。

| 文件 | 类型 | 是否自生成 | 生成/下载方案 | 当前用途 |
|---|---|---:|---|---|
| `data/train.json` | 内部训练源 | 是 | `scripts/generate_dataset.py` 调用 OpenAI 兼容 API 生成（用户已确认） | strict 训练池/验证池来源之一 |
| `data/val.json` | 内部训练源 | 是 | `scripts/generate_dataset.py` 调用 OpenAI 兼容 API 生成（用户已确认） | strict 训练池/验证池来源之一 |
| `data/ifeval_full_with_meta.json` | 内部 IF 训练源 | 是（本地模板 + API 响应） | `scripts/download_ifeval_datasets.py` 生成中文 IF 模板后，使用 `--generate-responses` 调用 API 生成响应 | strict 训练池/验证池 IF 来源之一（不用于测试） |
| `data/argilla_ifeval.json` | 公开 IF 训练源 | 否 | `scripts/download_ifeval_datasets.py` 下载/转换 `argilla/ifeval-like-data` | strict 训练池/验证池 IF 来源之一 |
| `data/public_eval_with_meta.json` | 公开翻译/总结数据（元信息保留） | 否 | `scripts/download_public_datasets.py` 下载并转换公开数据，本次实际来源标签为 `arxiv` / `opus-100` / `wmt19` | strict 公共数据切分上游源 |
| `data/public_train_strict_with_meta.json` | 公开切分源（训练） | 否（由公开数据切分） | `scripts/prepare_qwen3_strict_public_sources.py` 从 `public_eval_with_meta` 分层切分 | strict 训练池/验证池翻译/总结来源之一 |
| `data/public_val_strict_with_meta.json` | 公开切分源（验证预留） | 否（由公开数据切分） | 同上 | 当前默认不纳入训练（可选） |
| `data/public_test_strict_with_meta.json` | 公开切分源（测试） | 否（由公开数据切分） | 同上 | strict 有标注测试集翻译/总结来源 |
| `dataset/LRIA-Follow_EN/LRIA-Follow_v3_EN.xlsx` | 内部 IF 测试原始源 | 否（内部测试资产） | 项目内已有 xlsx，经转换提取 `Prompt/gt` | strict 有标注 IF 测试来源（经转换） |
| `dataset/LRIA-Follow_ZH/LRIA-Follow_v3_ZH.xlsx` | 内部 IF 测试原始源 | 否（内部测试资产） | 项目内已有 xlsx，经转换提取 `Prompt/gt` | strict 有标注 IF 测试来源（经转换） |
| `data/qwen3_strict_internal_if_labeled.json` | 内部 IF 测试转换产物 | 否（由内部测试源转换） | `scripts/convert_lria_follow_to_strict_test.py` | strict 有标注测试集 IF 来源 |
| `dataset/IFEval/input_data.jsonl` | 公开 prompt-only IF 测试源 | 否 | 项目内保存的 Google IFEval prompt-only 原始文件 | strict `test_if_unlabeled` 来源 |
| `dataset/m-ifeval/PMMEval-mifeval-*.json` | 公开 prompt-only IF 测试源 | 否 | 项目内保存的 PMMEval m-IFEval 多语种 prompt-only 文件 | strict `test_if_unlabeled` 来源 |
| `dataset/LRIATrans/*.json` | 内部辅助回放源 | 否（内部测试资产） | 项目内已有 JSON（`origin_prompt/prediction`） | 当前不纳入 strict 主评测（缺 reference） |

### 4.2.2 公开数据下载链接（建议用于审计复核）

本节给出“公开数据 -> 本地文件 -> 使用位置”的映射。建议在审计前同时补充：

- `data/source_registry_qwen3_strict*.json` 中的 `upstream_revision`
- 实际下载日期（`downloaded_at`）
- 对应文件哈希（可从 `data/audit/qwen3_strict_hashes.json` 回填）

#### A. 公开 IF 训练数据（Argilla）

- 数据集：`argilla/ifeval-like-data`
- 链接：`https://huggingface.co/datasets/argilla/ifeval-like-data`
- 本地文件：`data/argilla_ifeval.json`
- 下载脚本：`scripts/download_ifeval_datasets.py`
- 用途：strict 训练/验证 IF 来源之一

#### B. 公开 IF prompt-only 测试数据（IFEval）

- 数据集：`google/IFEval`
- 链接：`https://huggingface.co/datasets/google/IFEval`
- 本地文件（项目内保存）：`dataset/IFEval/input_data.jsonl`
- 相关脚本：`scripts/download_ifeval_datasets.py`（工程内也有 IFEval 下载逻辑）
- 用途：`qwen3_strict_test_if_unlabeled.json`

#### C. 公开翻译/总结数据（本次 strict 构建实际使用）

1. `arxiv` 摘要数据
- 数据集（脚本入口）：`ccdv/arxiv-summarization`
- 链接：`https://huggingface.co/datasets/ccdv/arxiv-summarization`
- 本地承载文件：`data/public_eval_with_meta.json`（切分前），`data/public_*_strict_with_meta.json`（切分后）
- 下载脚本：`scripts/download_public_datasets.py`
- 用途：strict 训练/验证/有标注测试（总结任务）

2. `opus-100` 翻译数据
- 数据集（本次实际成功加载）：`Helsinki-NLP/opus-100`
- 链接：`https://huggingface.co/datasets/Helsinki-NLP/opus-100`
- 本地承载文件：`data/public_eval_with_meta.json`（切分前），`data/public_*_strict_with_meta.json`（切分后）
- 下载脚本：`scripts/download_public_datasets.py`
- 用途：strict 训练/验证/有标注测试（翻译任务）

3. `wmt19` 翻译数据
- 数据集（本次实际成功加载）：`wmt/wmt19`
- 链接：`https://huggingface.co/datasets/wmt/wmt19`
- 本地承载文件：`data/public_eval_with_meta.json`（切分前），`data/public_*_strict_with_meta.json`（切分后）
- 下载脚本：`scripts/download_public_datasets.py`
- 用途：strict 训练/验证/有标注测试（翻译任务）

#### D. 公开多语种 IF prompt-only 测试数据（m-IFEval / PMMEval）

- 本地文件：`dataset/m-ifeval/PMMEval-mifeval-*.json`
- 用途：`qwen3_strict_test_if_unlabeled.json`
- 说明：当前工程中以本地文件形式存在并被直接读取；建议在最终审计版 `source_registry` 中补充其上游发布链接/仓库链接与版本信息（当前 README 先保留本地文件映射，避免写入未经你确认的链接）。

#### A. 用户确认的自生成训练源（API 生成）
- `data/train.json`（用户确认）
- `data/val.json`（用户确认）

用途：
- strict 训练池/验证池的一部分（不是固定只当 `val`）

#### B. 公开 IF 训练源
- `data/argilla_ifeval.json`
  - 来源：公开 IF 风格数据（Argilla IFEval-like）

#### C. 本地生成的中文 IF 训练源
- `data/ifeval_full_with_meta.json`
  - 性质：本地中文模板 + API 生成响应（不是纯公开原始数据）
  - 生成脚本：`scripts/download_ifeval_datasets.py`
  - 生成方案（当前工程口径）：
    1. 生成中文 IF 模板并写入 `data/ifeval_chinese_templates.json`
    2. 为模板记录打上 `source = chinese_generated`，初始 `output = ""`
    3. 使用 `--generate-responses` 调用 API 生成响应
    4. 将有响应样本保存为 `data/ifeval_full_with_meta.json`
  - 当前文件特征（已核查）：
    - 共 `579` 条
    - `source` 全部为 `chinese_generated`
    - `output` 全部非空
  - 当前 strict 用途（重要）：
    - 用于 `train_v1a` / `train_v1b` / `train_v1c` / `val`
    - 不用于 `test_labeled` 与 `test_if_unlabeled`

#### D. 公开翻译/总结源（metadata 保留）
- `data/public_eval_with_meta.json`
  - 由 `scripts/download_public_datasets.py` 下载生成
  - 当前包含来源（本次实际下载结果）：
    - `arxiv`
    - `opus-100`
    - `wmt19`
  - 公开来源说明（按本次脚本实际成功加载口径）：
    - `arxiv`：公开摘要数据（脚本中的 arXiv summarization 下载入口）
    - `opus-100`：公开翻译数据（本次实际从 `Helsinki-NLP/opus-100` 成功加载）
    - `wmt19`：公开翻译基准数据（本次实际从 `wmt/wmt19` 成功加载）
- 再由 `scripts/prepare_qwen3_strict_public_sources.py` 切分为：
  - `data/public_train_strict_with_meta.json`
  - `data/public_val_strict_with_meta.json`
  - `data/public_test_strict_with_meta.json`

#### E. 内部有标注 IF 测试源（来自 `dataset/`）
- `dataset/LRIA-Follow_EN/LRIA-Follow_v3_EN.xlsx`
- `dataset/LRIA-Follow_ZH/LRIA-Follow_v3_ZH.xlsx`

经脚本转换为：
- `data/qwen3_strict_internal_if_labeled.json`

#### F. 公开 prompt-only IF 测试源（来自 `dataset/`）
- `dataset/IFEval/input_data.jsonl`
- `dataset/m-ifeval/PMMEval-mifeval-*.json`

用于：
- `data/qwen3_strict_test_if_unlabeled.json`

### 4.3 当前未作为 strict 主评测输入的内部文件

- `dataset/LRIATrans/*.json`

原因：
- 当前文件结构为 `origin_prompt + prediction + response_time`
- 不含明确 reference output，不能作为“有标注翻译测试集”主证据
- 可作为历史模型回放/对照辅助数据

---

## 5. 当前 strict 数据集构建结果（实际统计）

数据来源：`data/qwen3_strict_manifest.json`

### 5.1 规模（counts）

- `train_v1a = 3600`
- `train_v1b = 3600`
- `train_v1c = 3600`
- `val = 360`
- `test_labeled = 900`
- `test_if_unlabeled = 1200`

### 5.2 任务分布（task distribution）

#### `v1a`（平衡）
- IF: 1200
- Translation: 1200
- Summarization: 1200

#### `v1b`（IF 优先）
- IF: 1800
- Translation: 900
- Summarization: 900

#### `v1c`
- 与 `v1b` 数据分布一致（当前数据复制）

#### 验证集 `val`
- IF: 120
- Translation: 120
- Summarization: 120

#### 有标注测试 `test_labeled`
- IF: 300
- Translation: 300
- Summarization: 300

#### 无标注 IF 测试 `test_if_unlabeled`
- IF: 1200（全部为 IF prompt-only）

### 5.3 来源分布（source distribution）

#### `train_v1a`
- `self_generated`: 1241
- `argilla_ifeval`: 882
- `arxiv`: 521
- `opus-100`: 432
- `chinese_generated`: 318
- `wmt19`: 206

#### `val`
- `self_generated`: 130
- `argilla_ifeval`: 79
- `arxiv`: 44
- `chinese_generated`: 41
- `opus-100`: 40
- `wmt19`: 26

#### `test_labeled`
- `arxiv`: 300
- `opus-100`: 194
- `lria_follow_zh`: 193
- `lria_follow_en`: 107
- `wmt19`: 106

#### `test_if_unlabeled`
来自 `IFEval + m-IFEval` 多语种（共 12 个来源标签）：
- `ifeval_prompt_only`: 357
- `mifeval_*`: 843（11 种语言合计）

### 5.4 池统计与可用性

当前 pool 统计显示（关键）：
- `missing_file_sources = []`（说明 `val.json` 已成功纳入）
- `labeled_test_pool_translation_en2zh = 268`
- `labeled_test_pool_translation_zh2en = 272`
- `labeled_test_pool_summarization = 360`
- `labeled_test_pool_instruction_following = 728`

这说明当前 `test_labeled=900` 的平衡抽样是由：
- 公共翻译/总结测试池
- 内部 LRIA-Follow IF 测试池
共同支持完成的。

### 5.5 数据隔离（Leakage）

当前 strict 构建结果显示以下全部为 `0`：

- `train_val`
- `train_test_labeled`
- `val_test_labeled`
- `train_test_if_unlabeled`
- `val_test_if_unlabeled`
- `test_labeled_vs_test_if_unlabeled`
- `train_v1b/v1c` 对 `val/test` 的交叉重叠

结论：
- 当前 `qwen3_strict` 构建满足严格数据隔离要求（以当前 dedup key 口径计）

---

## 6. `v1b / v1c` 的重复采样说明（必须披露）

`v1b` / `v1c` 是 IF 优先训练集（目标 50% IF），当前在 IF 样本池不足以支撑 `1800` 条唯一 IF 样本时，采用重复采样（with replacement）。

当前统计（`data/qwen3_strict_manifest.json`）：

- `duplicates_by_pool_shortage.instruction_following = 600`
- `duplicate_extra_instances = 600`
- `duplicate_groups = 486`

解释：
- 这是策略性重采样，不是构建错误
- 作用：增强 IF 学习信号
- 风险：模板过拟合风险上升，需在报告中透明说明

---

## 7. 审计工件（当前已生成）

目录：`data/audit/`

- `data/audit/qwen3_strict_lineage.jsonl`
  - 样本级血缘记录（split、source、source_file、source_index、dedup_key 等）
- `data/audit/qwen3_strict_source_snapshot.json`
  - 记录 direct input files（带大小）、final split 来源分布、lineage 数量
- `data/audit/qwen3_strict_hashes.json`
  - 输入/输出/脚本文件 SHA256 与大小

说明：
- `lineage` 中公共数据样本会保留上游 `_src_file = public_eval_with_meta.json`
- 这样可以追到“切分前的原始公开元信息文件”，而不是只停留在 split 文件

---

## 8. 执行命令（当前推荐流程，`granite_ft` 环境）

## 8.1 环境约定

微调与构建统一使用 `conda` 环境：

- `granite_ft`

说明：
- 后续命令均建议使用 `conda run -n granite_ft ...`

## 8.2 重新构建 strict 数据集（完整流程）

### Step 1：转换内部 LRIA-Follow 为 IF 有标注测试源

```powershell
conda run -n granite_ft python scripts/convert_lria_follow_to_strict_test.py
```

输出：
- `data/qwen3_strict_internal_if_labeled.json`
- `data/qwen3_strict_internal_if_manifest.json`

### Step 2：下载公开翻译/总结数据（带元信息）

```powershell
conda run -n granite_ft python scripts/download_public_datasets.py --output data/public_eval.json --arxiv-samples 1200 --translation-samples 1200
```

输出：
- `data/public_eval.json`
- `data/public_eval_with_meta.json`

### Step 3：切分公开数据为 strict 专用源文件（当前比例）

当前版本使用比例（为了支持 `test_labeled=900`）：
- train: `0.6`
- val: `0.1`
- test: `0.3`

```powershell
conda run -n granite_ft python scripts/prepare_qwen3_strict_public_sources.py --input data/public_eval_with_meta.json --train-ratio 0.6 --val-ratio 0.1 --test-ratio 0.3
```

输出：
- `data/public_train_strict_with_meta.json`
- `data/public_val_strict_with_meta.json`
- `data/public_test_strict_with_meta.json`
- `data/public_strict_split_manifest.json`

### Step 4：构建 strict 系列数据集（审计模式）

```powershell
conda run -n granite_ft python scripts/build_qwen3_strict_dataset.py --audit-mode --test-target 900
```

输出：
- `data/qwen3_strict_train.json` (`v1a`)
- `data/qwen3_strict_train_v1b.json`
- `data/qwen3_strict_train_v1c.json`
- `data/qwen3_strict_val.json`
- `data/qwen3_strict_test_labeled.json`
- `data/qwen3_strict_test_if_unlabeled.json`
- `data/qwen3_strict_manifest.json`
- `data/audit/qwen3_strict_*`

---

## 9. 训练与评测命令（strict v1a/v1b/v1c）

### 9.1 单次训练（非 3-seed）

`v1a`:
```powershell
conda run -n granite_ft llamafactory-cli train configs/finetune_qwen3_lora_strict_v1a.yaml
```

`v1b`:
```powershell
conda run -n granite_ft llamafactory-cli train configs/finetune_qwen3_lora_strict_v1b.yaml
```

`v1c`:
```powershell
conda run -n granite_ft llamafactory-cli train configs/finetune_qwen3_lora_strict_v1c.yaml
```

### 9.2 3-seed 训练（推荐）

默认 seeds：`2026, 2027, 2028`

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_3seed_train.ps1 -UseCondaRun
```

可选：只跑某个版本
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_3seed_train.ps1 -UseCondaRun -OnlyVersions v1b
```

可选：DryRun（仅生成 seed config 并检查命令）
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_3seed_train.ps1 -UseCondaRun -DryRun
```

### 9.3 3-seed 评测（strict 测试集）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_3seed_eval.ps1 -UseCondaRun
```

默认评测输入：
- `data/qwen3_strict_test_labeled.json`
- `data/qwen3_strict_test_if_unlabeled.json`

### 9.4 3-seed 结果聚合（均值/方差）

```powershell
conda run -n granite_ft python scripts/aggregate_qwen3_strict_3seed.py
```

输出：
- `evaluation/strict_3seed/aggregate_3seed_summary.json`

---

## 10. 关键源文件（当前版本）

### 10.1 strict 数据构建核心
- `scripts/build_qwen3_strict_dataset.py`
- `data/qwen3_strict_manifest.json`
- `data/audit/qwen3_strict_lineage.jsonl`
- `data/audit/qwen3_strict_source_snapshot.json`
- `data/audit/qwen3_strict_hashes.json`

### 10.2 内部 IF 测试转换
- `scripts/convert_lria_follow_to_strict_test.py`
- `data/qwen3_strict_internal_if_labeled.json`
- `data/qwen3_strict_internal_if_manifest.json`

### 10.3 公开数据准备
- `scripts/download_public_datasets.py`
- `scripts/prepare_qwen3_strict_public_sources.py`
- `data/public_eval_with_meta.json`
- `data/public_strict_split_manifest.json`

### 10.4 训练配置
- `configs/finetune_qwen3_lora_strict_v1a.yaml`
- `configs/finetune_qwen3_lora_strict_v1b.yaml`
- `configs/finetune_qwen3_lora_strict_v1c.yaml`

### 10.5 3-seed 脚手架
- `scripts/run_qwen3_strict_3seed_train.ps1`
- `scripts/run_qwen3_strict_3seed_eval.ps1`
- `scripts/aggregate_qwen3_strict_3seed.py`

---

## 11. 已知注意事项（当前版本）

1. `test_if_unlabeled` 是 prompt-only IF 测试集
- 不是有标准答案测试
- 评分结果依赖规则可检测覆盖率（需在最终报告说明）

2. `v1b/v1c` 包含 IF 重复采样
- 当前为 600 条额外重复实例（审计已记录）

3. `v1c` 当前数据与 `v1b` 相同
- 差异来自训练超参，不是数据

4. `data/source_registry_qwen3_strict.template.json` 仍为模板
- 其中 `license`、`sha256`、部分 `model_name/generated_at` 等字段需要在提交审计前补齐
- 可结合 `data/audit/qwen3_strict_hashes.json` 回填哈希

5. `dataset/LRIATrans/*.json` 当前未纳入主评测
- 如后续需要用于正式评测，必须补充 reference 或明确其角色（回放/对照）

6. `data/ifeval_full_with_meta.json` 需要明确标注为“内部生成数据”
- 它不是公开原始中文 IF 数据
- 它是本地中文模板 + API 生成响应的数据
- 当前仅用于训练/验证，不用于测试

## 11.1 FAQ：`data/ifeval_full_with_meta.json` 用在了哪里？是不是我自己生成的？

结论：

1. 用途（当前 `qwen3_strict`）：
- 仅用于训练/验证来源（IF 类样本的一部分）
- 不用于测试集

2. 当前实际使用量（见 `data/qwen3_strict_manifest.json`）：
- `train_v1a`：`chinese_generated = 318`
- `train_v1b`：`chinese_generated = 483`
- `train_v1c`：`chinese_generated = 483`
- `val`：`chinese_generated = 41`
- `test_labeled`：`0`
- `test_if_unlabeled`：`0`

3. 是否属于你自己生成的数据：
- 按当前工程脚本逻辑和文件内容证据，答案是：**是（本地生成模板 + API 响应）**
- 不应表述为“纯公开下载数据”

4. 为什么在 `v1b/v1c` 中数量更高：
- `v1b/v1c` 是 IF 优先训练集（50% IF）
- 会对 IF 样本重复采样，因此 `chinese_generated` 会随 IF 样本一起被放大

---

## 12. 推荐的下一步（在本 README 基础上）

1. 补全 `data/source_registry_qwen3_strict.template.json`（建议另存为正式版）
2. 启动 `v1a/v1b/v1c` 的 3-seed 训练
3. 跑 strict 3-seed 评测并聚合均值/方差
4. 基于聚合结果输出正式评估报告（管理层版 + 技术版）

---

## 13. 口径建议（汇报时）

建议固定使用以下表述，避免歧义：

- “本轮 `qwen3_strict` 使用审计级数据构建流程，区分了直接实验输入与上游来源。”
- “内部 `dataset/` 作为测试来源池的一部分，配合公开有标注测试源共同构成严格测试集。”
- “最终结论将基于 `qwen3_strict` 的 3-seed 结果（mean/std），而不是单次探索结果。”

---

## 14. 方案A（双 Adapter 派生路线）补充说明（与混合 Adapter 路线并行）

本节记录在 `qwen3_strict` 母数据集基础上新增的“双 Adapter”路线（方案A），用于缓解 1.7B 单 LoRA 同时学习翻译/总结/IF 的任务竞争问题。

### 14.1 路线目标与并行关系

当前并行推进两条路线：

1. 混合 Adapter 路线（保留）
- `v1a`（平衡）
- `v1b`（IF 优先）
- `v1c`（IF 优先 + 更低 LR）
- 用于形成主线对比与 3-seed 报告

2. 方案A 双 Adapter 路线（新增）
- `TS adapter`：翻译 + 总结（Translation + Summarization）
- `IF adapter`：指令遵循（Instruction Following）
- 在同一套 strict 测试集上进行“单 Adapter 能力评估”和“按任务路由后的系统级评估”

说明：
- 两条路线共享同一 `qwen3_strict` 测试口径（`test_labeled` + `test_if_unlabeled`）
- 方案A 不替代当前 `v1a/v1b/v1c`，而是并行探索并验证是否更适合 1.7B 容量约束

### 14.2 派生数据集（完全来源于 strict 母数据集 train/val）

派生脚本：
- `scripts/build_qwen3_strict_dual_adapter_datasets.py`

派生输入（母数据集）：
- `data/qwen3_strict_train.json`
- `data/qwen3_strict_val.json`

派生输出（当前版本）：
- `data/qwen3_strict_train_ts_v1.json`（TS 训练集，2400）
- `data/qwen3_strict_val_ts_v1.json`（TS 验证集，240）
- `data/qwen3_strict_train_if_v1.json`（IF 训练集，1200）
- `data/qwen3_strict_val_if_v1.json`（IF 验证集，120）
- `data/qwen3_strict_dual_adapter_manifest_v1.json`

分流规则：
- `translation` + `summarization` -> `TS adapter`
- `instruction_following` -> `IF adapter`

审计说明：
- 派生训练/验证集仅做任务分流，不引入新的外部样本源
- `data/qwen3_strict_dual_adapter_manifest_v1.json` 中记录了 parent 文件与 SHA256（用于复核）

### 14.3 LlamaFactory 数据集注册（方案A）

已注册到 `data/dataset_info.json`：
- `qwen3_strict_train_ts_v1`
- `qwen3_strict_val_ts_v1`
- `qwen3_strict_train_if_v1`
- `qwen3_strict_val_if_v1`

### 14.4 方案A 训练配置（1-seed 起步）

训练配置文件：
- `configs/finetune_qwen3_lora_strict_ts_v1.yaml`
- `configs/finetune_qwen3_lora_strict_if_v1.yaml`
- 训练脚本（新增，支持 1-seed / 3-seed）：
  - `scripts/run_qwen3_strict_planA_train.ps1`

设计依据（当前版本）：
- 在你已验证的 strict 配置上做“可比性优先”的小幅调整
- `TS adapter` 更接近 `v1a`（稳定基线）
- `IF adapter` 更接近 `v1c` 思路（更低 LR，降低 IF 过拟合风险）

#### 14.4.1 直接单次训练（逐个 adapter，手工命令）

单次训练命令（当前在 `granite_ft` 环境）：

```powershell
llamafactory-cli train configs/finetune_qwen3_lora_strict_ts_v1.yaml
```

```powershell
llamafactory-cli train configs/finetune_qwen3_lora_strict_if_v1.yaml
```

如需显式指定 conda：

```powershell
conda run -n granite_ft llamafactory-cli train configs/finetune_qwen3_lora_strict_ts_v1.yaml
conda run -n granite_ft llamafactory-cli train configs/finetune_qwen3_lora_strict_if_v1.yaml
```

#### 14.4.2 使用方案A训练脚本（推荐，支持跳过已完成任务）

`scripts/run_qwen3_strict_planA_train.ps1` 的目标：
- 同时管理 `TS/IF` 两个 adapter 的训练
- 支持 `1-seed` 与 `3-seed`
- 自动检测是否已训练完成（`epoch >= target_epoch`）并跳过
- 在 `3-seed` 模式下自动生成 seed 配置文件

默认行为（不带 `-ThreeSeed`）：
- 运行 `TS` + `IF` 两个 adapter 的单次训练（使用基础 yaml）
- 输出目录分别为：
  - `outputs/qwen3-1.7B-lora-strict-ts-v1`
  - `outputs/qwen3-1.7B-lora-strict-if-v1`

1-seed（当前已激活 `granite_ft` 环境）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_train.ps1
```

可选：只跑某一个 adapter

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_train.ps1 -OnlyAdapters ts
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_train.ps1 -OnlyAdapters if
```

如需显式走 conda：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_train.ps1 -UseCondaRun
```

3-seed（自动生成 seed 配置并训练）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_train.ps1 -ThreeSeed
```

可选：指定种子 / 指定 adapter / DryRun

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_train.ps1 -ThreeSeed -Seeds 2026,2027,2028 -OnlyAdapters ts -DryRun
```

3-seed 模式生成配置目录：
- `configs/generated_strict_planA_3seed/`

---

## 15. `score.py` 升级后的脚本兼容性（Strict / Hybrid IF 评分口径）

本项目的 `scripts/score.py` 已进行较大升级，新增了来源感知 IF 评分能力与可选参数：

- `--enable-lria-fallback`

该参数用于在原有规则型 IF 评分（strict 口径）基础上，增加针对 `LRIA-Follow` 来源样本的 fallback 判定（hybrid 口径），主要目的是提升 `IF_labeled` 覆盖率。

### 15.1 哪些脚本需要改，哪些不需要改

结论：

1. 训练脚本通常不需要改
- 训练脚本（如 `scripts/run_qwen3_strict_3seed_train.ps1`、`scripts/run_qwen3_strict_planA_train.ps1`）不调用 `scripts/score.py`
- 因此不受 `score.py` 新参数影响

2. 评测脚本需要支持新评分口径（已完成）
- 因为评测脚本会调用 `scripts/generate.py` 和 `scripts/score.py`
- 为了后续“仍按同样方式使用脚本”，已在评测脚本中增加 `-HybridIF` 开关

已支持 `-HybridIF` 的脚本：

- `scripts/run_qwen3_strict_base_eval.ps1`
- `scripts/run_qwen3_strict_3seed_eval.ps1`
- `scripts/run_qwen3_strict_planA_eval.ps1`

### 15.2 `Strict` 与 `HybridIF` 两种评分口径

1. `Strict`（默认）
- 不传 `-HybridIF`
- `score.py` 不启用 `--enable-lria-fallback`
- 适合保持历史规则口径可比性

2. `HybridIF`
- 传 `-HybridIF`
- 脚本会自动给 `score.py` 增加 `--enable-lria-fallback`
- 主要提升 `IF_labeled` 覆盖率（`IF_unlabeled` 基本不受影响）

说明：
- `HybridIF` 不会重跑训练，也不会改变模型生成结果
- 只会影响评分阶段的 IF 指标与覆盖率统计口径
- 翻译/总结指标（BLEU/ROUGE/BERTScore）不受影响

### 15.3 评测脚本在 `HybridIF` 下的目录行为（已自动处理）

为避免覆盖 strict 结果，评测脚本在传入 `-HybridIF` 时会自动把评测输出目录改为带 `_hybrid` 后缀的版本：

- `evaluation/strict_3seed` -> `evaluation/strict_3seed_hybrid`
- `evaluation/strict_planA` -> `evaluation/strict_planA_hybrid`

`base` 脚本同理，默认 `EvalDir` 为 `evaluation/strict_3seed`，启用 `-HybridIF` 后会写入：

- `evaluation/strict_3seed_hybrid/base_labeled`
- `evaluation/strict_3seed_hybrid/base_if_unlabeled`

### 15.4 推荐用法（后续版本迭代 / 3-seed 通用）

#### A. Base 评测（Strict）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_base_eval.ps1
```

#### B. Base 评测（HybridIF）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_base_eval.ps1 -HybridIF
```

#### C. 混合 Adapter 3-seed 评测（Strict）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_3seed_eval.ps1
```

#### D. 混合 Adapter 3-seed 评测（HybridIF）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_3seed_eval.ps1 -HybridIF
```

#### E. 方案A（PlanA 路由评测，Strict）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_eval.ps1
```

#### F. 方案A（PlanA 路由评测，HybridIF）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_eval.ps1 -HybridIF
```

### 15.5 对既有生成结果做重评分（不重跑 `generate.py`）

如果你已经完成生成，只想用新 `score.py` 口径重评分，继续使用：

- `scripts/run_qwen3_strict_hybrid_rescore_260225.ps1`

示例（HybridIF 默认开启）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_hybrid_rescore_260225.ps1
```

可选 strict 口径：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_hybrid_rescore_260225.ps1 -StrictOnly
```

### 15.6 一键自动聚合（Strict / Hybrid / PlanA）

为避免每次手动给 `scripts/aggregate_qwen3_strict_3seed.py` 传目录与版本参数，新增统一聚合脚本：

- `scripts/run_qwen3_strict_auto_aggregate.ps1`

默认行为（存在即聚合，不存在则跳过）：

1. 混合 Adapter 3-seed（Strict）
- `evaluation/strict_3seed` -> `evaluation/strict_3seed/aggregate_3seed_summary.json`

2. 混合 Adapter 3-seed（HybridIF）
- `evaluation/strict_3seed_hybrid` -> `evaluation/strict_3seed_hybrid/aggregate_3seed_summary.json`

3. 方案A PlanA 3-seed（Strict）
- `evaluation/strict_planA` -> `evaluation/strict_planA/aggregate_tsif_3seed_summary.json`

4. 方案A PlanA 3-seed（HybridIF）
- `evaluation/strict_planA_hybrid` -> `evaluation/strict_planA_hybrid/aggregate_tsif_3seed_summary.json`

推荐用法（当前已激活 `granite_ft` 环境）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_auto_aggregate.ps1
```

显式使用 conda：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_auto_aggregate.ps1 -UseCondaRun
```

常用可选参数：

- `-Force`：覆盖已有聚合结果
- `-DryRun`：只打印将执行的命令
- `-Seeds 2026,2027,2028`：指定聚合 seed 列表
- `-MixedVersions v1a,v1b,v1c`：指定混合 Adapter 版本列表
- `-PlanAVersions tsif_v1`：指定 PlanA 版本列表

示例（只做命令预览）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_auto_aggregate.ps1 -DryRun
```

示例（强制重算）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_auto_aggregate.ps1 -Force
```

### 15.7 后续迭代建议（为了“同样方法继续用脚本”）

后续新增版本（例如 `v2*` 或方案A的 `tsif_v2*`）时，建议保持以下原则：

1. 训练脚本保持不变（除非训练流程本身变化）
2. 评测脚本统一保留 `-HybridIF` 开关
3. 聚合阶段优先使用 `scripts/run_qwen3_strict_auto_aggregate.ps1`
4. 报告中同时注明评分口径（`Strict` 或 `HybridIF`）
5. 如需做跨版本对比，尽量保证使用同一评分口径


命名示例：
- `configs/generated_strict_planA_3seed/finetune_qwen3_lora_strict_ts_v1_s2026.yaml`
- `configs/generated_strict_planA_3seed/finetune_qwen3_lora_strict_if_v1_s2026.yaml`

3-seed 输出目录命名示例：
- `outputs/qwen3-1.7B-lora-strict-ts-v1-s2026`
- `outputs/qwen3-1.7B-lora-strict-if-v1-s2026`

### 14.5 方案A 评测脚本（按任务路由评测，支持 1-seed / 3-seed）

方案A 路由评测相关脚本：
- `scripts/build_qwen3_strict_routed_eval_inputs.py`
- `scripts/merge_qwen3_strict_routed_labeled_outputs.py`
- `scripts/run_qwen3_strict_planA_eval.ps1`

路由策略（当前版本）：
- `test_labeled` 中：
  - `translation` / `summarization` -> `TS adapter`
  - `instruction_following` -> `IF adapter`
- `test_if_unlabeled`（1200 条 prompt-only IF）：
  - 全部由 `IF adapter` 生成并评分

#### 14.5.1 1-seed 路由评测（默认模式）

默认假设适配器目录为：
- `outputs/qwen3-1.7B-lora-strict-ts-v1`
- `outputs/qwen3-1.7B-lora-strict-if-v1`

运行命令（当前已激活 `granite_ft` 环境时推荐）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_eval.ps1
```

如需显式走 conda：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_eval.ps1 -UseCondaRun
```

默认输出：
- 生成中间结果：
  - `evaluation/output_data/planA_routing/*`（路由后的 labeled 子集与 manifest）
  - `evaluation/output_data/planA_eval/*`（TS/IF 生成结果、merge 后 labeled 结果）
- 评分结果：
  - `evaluation/strict_planA/tsif_v1_labeled/`
  - `evaluation/strict_planA/tsif_v1_if_unlabeled/`

#### 14.5.2 3-seed 路由评测（同一脚本，`-ThreeSeed` 模式）

默认 seeds：
- `2026, 2027, 2028`

默认命名约定（脚本自动拼接）：
- TS adapter：`outputs/qwen3-1.7B-lora-strict-ts-v1-s<seed>`
- IF adapter：`outputs/qwen3-1.7B-lora-strict-if-v1-s<seed>`

运行命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_eval.ps1 -ThreeSeed
```

如需显式走 conda：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_eval.ps1 -ThreeSeed -UseCondaRun
```

可选：指定种子 / 自定义版本标签：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_eval.ps1 -ThreeSeed -Seeds 2026,2027,2028 -VersionTag tsif_v1
```

3-seed 输出目录命名示例：
- `evaluation/strict_planA/tsif_v1_s2026_labeled/`
- `evaluation/strict_planA/tsif_v1_s2026_if_unlabeled/`
- `evaluation/strict_planA/tsif_v1_s2027_*`
- `evaluation/strict_planA/tsif_v1_s2028_*`

#### 14.5.3 方案A 3-seed 聚合（复用现有聚合脚本）

可直接复用 `scripts/aggregate_qwen3_strict_3seed.py`，只需指定 `eval-dir` 和 `versions`：

```powershell
conda run -n granite_ft python scripts/aggregate_qwen3_strict_3seed.py --eval-dir evaluation/strict_planA --versions tsif_v1 --output evaluation/strict_planA/aggregate_tsif_v1_3seed_summary.json
```

### 14.6 方案A 与混合 Adapter 路线的并行推进建议（当前阶段）

你当前可并行执行：

1. 混合 Adapter 主线（`v1a/v1b/v1c`）
- 继续完成 3-seed 评测
- 补 Base strict 基线
- 聚合 mean/std 后形成主对比表

2. 方案A 双 Adapter 路线
- 先跑 `TS/IF` 各 1-seed 训练 + 路由评测（方向验证）
- 若方向成立，再上方案A 的 3-seed 训练与路由评测

推荐先后顺序（资源有限时）：
1. 完成混合 Adapter 路线 3-seed 评测（主线）
2. 完成方案A 1-seed 路由评测（方向验证）
3. 再决定是否投入方案A 3-seed

建议命令组合（并行推进时）：

1. 方案A 1-seed 训练（两个 adapter）
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_train.ps1
```

2. 方案A 1-seed 路由评测
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_eval.ps1
```

3. 方案A 3-seed 训练（待方向确认后）
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_train.ps1 -ThreeSeed
```

4. 方案A 3-seed 路由评测（待方向确认后）
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_qwen3_strict_planA_eval.ps1 -ThreeSeed
```

### 14.7 方案A 的当前边界（需要在后续报告中说明）

1. 方案A 当前已实现“路由后系统级评测”脚本，但尚未替代混合 Adapter 主路线
2. 复合请求（同时要求翻译/总结 + 严格格式约束）未来可能需要更细路由策略或 fallback 方案
3. 方案A 当前阶段的目标是验证“拆分 Adapter 是否在 1.7B 上缓解任务竞争”，不是立即取代所有混合方案
