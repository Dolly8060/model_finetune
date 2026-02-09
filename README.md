# Granite 4.0-1B 微调工程

基于 LLaMA-Factory 的模型微调工程，专注于提升 Granite 4.0-1B 在**中英翻译**和**文章总结**任务上的能力。

## 核心功能

| 模块 | 功能 |
|------|------|
| 数据生成 | 调用AI API生成高质量翻译+总结混合数据集（聚焦计算机/AI领域） |
| 模型微调 | 支持LoRA/QLoRA/全量微调，YAML配置驱动 |
| 效果评估 | BLEU/ROUGE/BERTScore + 指令遵循约束检测(35种约束模式)，两阶段评估(推理+评分分离) |

## 数据生成设计原则

针对1B小模型的特点，数据生成遵循三大原则：

### A. 指令多样性 (Instruction Diversity)
避免所有指令都一模一样导致模型"背板"，每种任务使用多种不同的指令表述：

| 任务 | 指令示例 |
|------|----------|
| 英译中 | "将下面这段计算机领域的学术摘要翻译成中文。"、"Translate to Chinese:"、"中文翻译：" |
| 中译英 | "请把这段中文翻译成英文。"、"English translation:"、"Translate to English:" |
| 总结 | "请提取核心观点，并进行简要总结。"、"用一句话概括其主要贡献。"、"Summarize the key points." |

### B. 数据配比 (Data Balancing)
保持任务数据量平衡，避免模型忽略某个任务：

| 配比参数 | 翻译:总结 | 适用场景 |
|----------|-----------|----------|
| `--translation-ratio 0.5` | 1:1 | 默认推荐，两个任务均衡 |
| `--translation-ratio 0.67` | 2:1 | 侧重翻译能力 |

### C. 统一领域 (Domain Consistency)
聚焦**计算机/AI领域**，避免跨领域导致的灾难性遗忘：

- 深度学习：Transformer架构、注意力机制、模型压缩、LoRA微调
- NLP：文本生成、机器翻译、Prompt Engineering
- 系统工程：分布式训练、边缘计算推理、MLOps
- 前沿研究：RLHF、自监督学习、模型可解释性

# v5 模型版本说明

## 版本定位

**当前版本**: v5 (基于 v3 数据集训练的生产版本)

- **训练数据**: `train_v3.json` (翻译:总结≈1.7:1,中英双向平衡)
- **验证数据**: `val_v3.json` (用于 checkpoint 选择)
- **测试数据**: `test_v3.json` (最终性能评估)
- **输出模型**: `outputs/granite-4.0-1B-lora_v5_translation`

## v5 性能指标 (基于 test_v3.json)

### 翻译任务效果

| 指标 | 基础模型 | v5 微调模型 | 提升幅度 | 业务解读 |
|------|----------|-------------|----------|----------|
| **BLEU** | 30.33 | 39.69 | **+9.36** | 翻译准确度提升 30.9%,达到实用级别 |
| **ROUGE-1** | 48.92 | 57.14 | +8.22 | 词汇覆盖率提升,信息保留更完整 |
| **ROUGE-2** | 30.11 | 40.07 | **+9.97** | 短语级对齐能力显著增强 |
| **ROUGE-L** | 45.54 | 53.91 | +8.37 | 句子结构相似度提升,流畅度改善 |
| **BERTScore-F1** | 92.81 | 93.80 | +0.99 | 语义一致性达到专业翻译水平 |

**核心结论**: v5 在翻译任务上 BLEU 突破 39,相比基础模型提升 30.9%,ROUGE-2 提升 33.1%,已达到生产可用标准。

### 总结任务效果

| 指标 | 基础模型 | v5 微调模型 | 提升幅度 | 业务解读 |
|------|----------|-------------|----------|----------|
| **ROUGE-1** | 16.38 | 34.91 | **+18.53** | 关键词召回率提升 113%,抓取核心信息能力大幅增强 |
| **ROUGE-2** | 6.71 | 14.06 | +7.35 | 短语级摘要能力提升 109% |
| **ROUGE-L** | 14.60 | 30.09 | **+15.49** | 摘要结构化能力提升 106% |
| **BERTScore-F1** | 78.50 | 87.12 | **+8.62** | 语义保留度提升至 87.12,接近人工摘要水平 |

**核心结论**: v5 在总结任务上实现质的飞跃,ROUGE-1/L 提升超 100%,BERTScore-F1 达到 87.12,已具备生产级摘要能力。

## v3 数据集架构

### 数据文件角色划分

| 角色 | 文件 | 样本数 | 用途 |
|------|------|--------|------|
| **训练集** | `train_v3.json` | ~3400 条 | v5 模型训练数据源,翻译:总结≈1.7:1 |
| **验证集** | `val_v3.json` | ~400 条 | 与训练集完全隔离,用于 checkpoint 选择 |
| **测试集** | `test_v3.json` | ~200 条 | 与训练/验证集完全隔离,v5 性能指标来源 |
| **公开评估集** | `public_val_v2.json` | 900 条 | HuggingFace 公开数据,用于对外可复现基准 |
| **历史数据** | `train.json` / `train_mixed_3k.json` | - | 早期版本训练数据,已被 v3 数据集替代 |

### 评估策略

#### 推荐评估流程

1. **训练中验证**: `data/val_v3.json` - 用于选择最优 checkpoint
2. **最终性能测试**: `data/test_v3.json` - v5 版本的官方指标来源(见上表)
3. **公开基准对比**: `data/public_val_v2.json` - 用于与其他模型横向对比
4. **分方向诊断**: `public_val_v2_en2zh.json` / `public_val_v2_zh2en.json` - 诊断英→中/中→英性能差异

#### 评估子集自动拆分

`scripts/evaluate.py` 在加载任意评估文件后,会自动按 **任务类型** 拆分:

- **翻译子集 (translation)**: 通过 `task_type` 或指令关键词("翻译/translate/译为")识别
  - 主要指标: **BLEU + ROUGE-1/2/L + BERTScore-F1**
  - 解读重点: BLEU 衡量准确度,ROUGE 衡量信息保留,BERTScore 衡量语义一致性

- **总结子集 (summarization)**: 通过 `task_type` 或指令关键词("总结/摘要/概括/summary")识别
  - 主要指标: **ROUGE-1/2/L + BERTScore-F1** (BLEU 仅作参考)
  - 解读重点: ROUGE-1 看关键词召回,ROUGE-L 看结构保留,BERTScore 看语义完整度

**参数说明**:
- `--max-samples 0` (默认): 全量评估,确保结果可靠性
- `--max-samples 100`: 快速验证,仅用于开发调试

## 目录结构

```
model_finetune/
├── configs/                    # 微调配置文件
│   ├── finetune_lora_v2.yaml  # v5 版本配置(推荐,翻译+总结增强)
│   ├── finetune_lora.yaml     # 基础 LoRA 配置
│   ├── finetune_qlora.yaml    # QLoRA 低显存配置(4-bit 量化)
│   └── finetune_full.yaml     # 全量微调配置(需 8GB+ 显存)
├── data/                       # 数据目录
│   ├── dataset_info.json      # LLaMA-Factory 数据集注册文件
│   ├── train_v3.json          # v3 训练集(v5 模型训练数据源)
│   ├── val_v3.json            # v3 验证集(checkpoint 选择)
│   ├── test_v3.json           # v3 测试集(v5 性能指标来源)
│   ├── test_v4_enhanced.json  # v4 增强测试集(翻译+总结+指令遵循, 1643条)
│   ├── public_val_v2.json     # 公开评估集(900 条,0% 训练集重叠)
│   ├── public_val_v2_en2zh.json   # 英→中翻译子集(284 条)
│   ├── public_val_v2_zh2en.json   # 中→英翻译子集(316 条)
│   ├── train.json             # 历史训练数据(已被 train_v3.json 替代)
│   └── val.json               # 历史验证数据
├── scripts/
│   ├── generate_dataset.py    # 数据集生成脚本(支持 API 批量生成)
│   ├── download_public_datasets.py  # 公开数据集下载工具
│   ├── build_v3_dataset.py    # v3 数据集构建脚本
│   ├── build_test_v4_enhanced.py   # v4 增强测试集构建(翻译+总结+IF)
│   ├── augment_training_data.py    # 数据增强与混合工具
│   ├── analyze_dataset.py     # 数据集统计分析工具
│   ├── evaluate.py            # 模型评估脚本(支持多模型对比)
│   ├── generate.py            # 批量推理脚本(多模型推理并保存结果)
│   ├── score.py               # 评分脚本(指标计算+指令遵循约束检测)
│   └── monitor_training.py    # 训练进度实时监控工具
├── evaluation/                 # 评估结果输出目录
│   ├── output_data/           # generate.py 推理结果
│   ├── performance/           # score.py 评分报告
│   └── ...                    # 其他历史评估结果
├── outputs/                    # 微调模型输出目录(被 .gitignore 忽略)
│   └── granite-4.0-1B-lora_v5_translation/  # v5 版本 LoRA 适配器
├── .env                        # 环境配置文件(被 .gitignore 忽略)
├── .env.example               # 环境配置模板
├── .gitignore                 # Git 忽略配置
├── environment.yml            # conda 环境定义文件
├── requirements.txt           # pip 依赖列表
├── run.py                     # 统一入口脚本
├── run.bat                    # Windows 启动脚本
└── run.sh                     # Linux/Mac 启动脚本
```

## 环境配置

### 1. 创建conda环境

```bash
cd d:/AI_code/model_finetune
conda create -n granite_finetune python=3.10 -y
conda activate granite_finetune
```

### 2. 安装PyTorch（CUDA版本）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件（注意：值不需要加引号）：

```ini
# API配置（用于生成数据集）
API_BASE_URL=https://api.deepseek.com/v1
API_KEY=sk-xxxxxxxxxxxxxxxx
API_MODEL=deepseek-chat

# 模型路径
MODEL_PATH=D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16
```

**支持的API服务：**
| 服务 | API_BASE_URL | API_MODEL |
|------|--------------|-----------|
| OpenAI | https://api.openai.com/v1 | gpt-4o, gpt-4-turbo |
| DeepSeek | https://api.deepseek.com/v1 | deepseek-chat |
| 通义千问 | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-plus, qwen-turbo |
| Gemini | 需通过兼容代理 | gemini-1.5-pro |

## 运行流程

### 下载公开评估数据集（可选）

如果不想调用API生成数据，可以直接下载公开数据集用于评估或微调：

```bash
conda activate granite_ft
python scripts/download_public_datasets.py
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | all | 下载的数据集：`all`/`arxiv`/`wmt`/`flores` |
| `--arxiv-samples` | 500 | arXiv总结样本数（0=全部） |
| `--translation-samples` | 500 | 翻译样本数（0=全部） |
| `--output` | data/public_val_v2.json | 输出文件路径 |
| `--all` | - | 下载全部数据（不限制样本数） |

**使用示例：**
```bash
# 下载所有数据集，各500条样本
python scripts/download_public_datasets.py

# 仅下载翻译数据集，200条样本
python scripts/download_public_datasets.py --dataset wmt --translation-samples 200

# 仅下载arXiv总结数据集，全部数据
python scripts/download_public_datasets.py --dataset arxiv --arxiv-samples 0
```

**数据来源：**
| 任务 | 数据集 | 说明 |
|------|--------|------|
| 总结 | ccdv/arxiv-summarization | arXiv论文摘要数据集 |
| 翻译 | Helsinki-NLP/opus-100 | OPUS多语言平行语料 |
| 翻译 | wmt/wmt19 | WMT19新闻翻译数据集 |

**输出位置：**
- `data/public_val_v2.json` - Alpaca格式评估数据

### （可选）查看数据集统计

```bash
conda activate granite_ft
python scripts/analyze_dataset.py data/train_mixed_3k.json
```

该脚本会输出：指令分布、翻译/总结任务占比、输入/输出长度分布，以及若干样本示例，便于检查数据质量。

### 步骤1：生成训练数据

```bash
conda activate granite_finetune
python run.py generate --total 2000 --translation-ratio 0.5
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total` | 2000 | 生成样本总数 |
| `--translation-ratio` | 0.5 | 翻译数据占比（0.5=1:1配比，0.67=2:1配比） |
| `--output` | data/train.json | 输出文件路径 |
| `--workers` | 5 | 并行请求线程数 |

**输出位置：**
- `data/train.json` - 训练集（90%）
- `data/val.json` - 验证集（10%）
- `data/full_with_meta.json` - 完整数据（含元信息，用于分析）

**生成数据示例：**
```json
[
  {
    "instruction": "将下面这段计算机领域的学术摘要翻译成中文。",
    "input": "We introduce a novel attention mechanism that reduces computational complexity...",
    "output": "我们引入了一种新颖的注意力机制，它降低了计算复杂度..."
  },
  {
    "instruction": "阅读下文，用一句话概括其主要贡献。",
    "input": "Current methods require massive labeled data. Our approach utilizes self-supervised learning to...",
    "output": "本文的主要贡献在于利用自监督学习减少了对大量标注数据的依赖。"
  }
]
```

### 步骤2：微调模型

```bash
python run.py train --config configs/finetune_lora_v2.yaml
```

#### （可选）训练过程监控

在另一个终端窗口中实时监控训练进度和检查点：

```bash
cd d:/AI_code/model_finetune
conda activate granite_ft
python scripts/monitor_training.py --output-dir outputs/granite-4.0-1B-lora_v5_translation --interval 30
```

该脚本会周期性打印当前 step/epoch、loss、学习率、预计剩余时间以及已保存的 checkpoint 列表。

**配置选择：**
| 配置文件 | 显存需求 | 适用场景 |
|----------|----------|----------|
| `finetune_lora_v2.yaml` | 4-6GB | v5版本，翻译+总结增强版（推荐） |
| `finetune_lora.yaml` | 4-6GB | 基础LoRA配置 |
| `finetune_qlora.yaml` | 2-4GB | 显存受限环境 |
| `finetune_full.yaml` | 8-12GB | 追求最佳效果 |

**关键训练参数（在yaml中配置）：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_rank` | 16 | LoRA秩，越大容量越强 |
| `learning_rate` | 2e-4 | 学习率 |
| `num_train_epochs` | 3.0 | 训练轮数 |
| `per_device_train_batch_size` | 4 | 单卡batch大小 |

**输出位置：**
- `outputs/granite-4.0-1B-lora_v5_translation/` - v5 LoRA适配器（finetune_lora_v2.yaml）
- `outputs/granite-4.0-1B-full/` - 全量微调模型（如使用finetune_full.yaml）

### 步骤3：评估对比

支持三种评估模式：微调前后对比、多模型横向对比、两阶段评估（推理+评分分离）。

#### 模式A：微调前后对比

```bash
# 验证集评估（v3验证集）
python scripts/evaluate.py \
  --base-model D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16 \
  --adapter outputs/granite-4.0-1B-lora_v5_translation \
  --eval-file data/val_v3.json \
  --output-dir evaluation/val_v5

# 公开数据评估（public_val_v2）
python scripts/evaluate.py \
  --base-model D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16 \
  --adapter outputs/granite-4.0-1B-lora_v5_translation \
  --eval-file data/public_val_v2.json \
  --output-dir evaluation/public_v5
```

> 说明：不指定 `--max-samples` 时，将对文件中的全部样本进行评估。

**参数说明：**
| 参数 | 说明 |
|------|------|
| `--base-model` | 原始模型路径 |
| `--adapter` | LoRA适配器路径（与--finetuned-model二选一） |
| `--finetuned-model` | 全量微调模型路径 |
| `--eval-file` | 评估数据文件 |
| `--max-samples` | 最大评估样本数（默认0=全量评估，仅在需要加速时设置为较小值） |
| `--output-dir` | 评估结果输出目录（默认为 `evaluation`） |

**输出位置：**
- `evaluation/eval_report.md` - Markdown格式评估报告
- `evaluation/eval_results.json` - JSON格式指标数据（按 translation / summarization 子集拆分）

#### （可选）翻译分方向评估

```bash
# 英→中翻译评估
python scripts/evaluate.py \
  --base-model D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16 \
  --adapter outputs/granite-4.0-1B-lora_v5_translation \
  --eval-file data/public_val_v2_en2zh.json \
  --output-dir evaluation/en2zh_v5

# 中→英翻译评估
python scripts/evaluate.py \
  --base-model D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16 \
  --adapter outputs/granite-4.0-1B-lora_v5_translation \
  --eval-file data/public_val_v2_zh2en.json \
  --output-dir evaluation/zh2en_v5
```

#### 模式B：多模型横向对比

对比微调后的小模型与更大规模模型的表现差异：

```bash
conda activate granite_ft
python scripts/evaluate.py --compare-models \
  "微调Granite-1B:D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16:outputs/granite-4.0-1B-lora_v5_translation" \
  "Qwen2.5-7B:Qwen/Qwen2.5-7B-Instruct" \
  "LLaMA-3.1-8B:meta-llama/Llama-3.1-8B-Instruct" \
  --eval-file data/val.json \
  --max-samples 50 \
  --output-dir evaluation/compare
```

**模型配置格式：**
```
"显示名称:模型路径"                    # 无adapter
"显示名称:模型路径:adapter路径"         # 带adapter

# Windows路径示例（支持 D:/... 格式）：
"微调模型:D:/models/base:D:/adapters/lora"
```

**使用示例：**
```bash
# 对比本地模型与HuggingFace Hub模型
python scripts/evaluate.py --compare-models \
  "基础Granite-1B:D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16" \
  "微调Granite-1B:D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16:outputs/granite-4.0-1B-lora_v5_translation" \
  "Qwen2.5-7B:Qwen/Qwen2.5-7B-Instruct" \
  "LLaMA-3.1-8B:meta-llama/Llama-3.1-8B-Instruct" \
  --eval-file data/public_val_v2.json \
  --max-samples 100 \
  --output-dir evaluation/compare

# 仅对比HuggingFace Hub上的模型
python scripts/evaluate.py --compare-models \
  "Qwen2.5-7B:Qwen/Qwen2.5-7B-Instruct" \
  "LLaMA-3.1-8B:meta-llama/Llama-3.1-8B-Instruct" \
  --eval-file data/public_val_v2.json
```

**支持的模型类型（自动检测prompt模板）：**
| 模型系列 | 示例 |
|----------|------|
| Granite | `ibm-granite/granite-3.0-1b-a400m-instruct` |
| Qwen | `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct` |
| LLaMA | `meta-llama/Llama-3.1-8B-Instruct` |
| Gemma | `google/gemma-2-9b-it` |

**输出位置：**
- `evaluation/comparison_report.md` - 多模型对比报告
- `evaluation/comparison_results.json` - 各模型指标数据
- `evaluation/comparison_details.json` - 详细输出样本（每模型前10条）

#### 模式C：两阶段评估（推理+评分分离） — 推荐

将推理和评分分开执行，适用于：
- 推理耗时长，需要分批运行
- 想复用推理结果进行多次评分（调整约束检测后无需重新推理）
- 需要检查推理输出后再评分
- 需要评估**指令遵循 (Instruction Following)** 能力

##### 第一步：批量推理 (generate.py)

对一个或多个模型运行推理，保存预测结果到 JSON 文件。

```bash
conda activate granite_ft

# 单模型推理（推荐使用 test_v4_enhanced.json，包含翻译+总结+指令遵循）
python scripts/generate.py \
  --models "Granite-v6:D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16:outputs/granite-4.0-1B-lora_v6_instruction" \
  --eval-file data/test_v4_enhanced.json

# 单模型推理（自定义输出路径）
python scripts/generate.py \
  --models "Granite-v6:D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16:outputs/granite-4.0-1B-lora_v6_instruction" \
  --eval-file data/test_v4_enhanced.json \
  --output-file evaluation/output_data/my_results.json

# 多模型推理
python scripts/generate.py \
  --models \
    "Base-Granite:D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16" \
    "Granite-v6:D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16:outputs/granite-4.0-1B-lora_v6_instruction" \
    "Qwen3-4B:D:/AI_code/models/Qwen3-4B-Instruct-2507" \
  --eval-file data/test_v4_enhanced.json \
  --max-samples 50
```

**generate.py 参数说明：**

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--models` | 是 | - | 模型配置列表，格式见下方 |
| `--eval-file` | 是 | - | 评估数据文件路径 |
| `--output-dir` | 否 | `evaluation/output_data` | 输出目录 |
| `--output-file` | 否 | - | 自定义输出路径（仅单模型有效，覆盖 `--output-dir`） |
| `--max-samples` | 否 | 0 | 最大样本数，0=全部 |
| `--max-new-tokens` | 否 | 512 | 生成的最大 token 数 |

**模型配置格式：**
```
"显示名称:模型路径"                    # 无 adapter
"显示名称:模型路径:adapter路径"         # 带 adapter（LoRA）

# Windows 路径示例：
"Granite-v6:D:/models/granite-1b:D:/outputs/lora_v6"
```

**输出位置：**
- `evaluation/output_data/{模型名}_{时间戳}.json` - 推理结果文件

##### 第二步：评分 (score.py)

对单个推理结果文件进行评分。score.py 会自动：
1. **按任务类型分组**：翻译 / 总结 / 指令遵循 / 其他
2. **运行时重分类**：修正数据集中的分类错误（无需重建数据集）
3. **计算内容质量指标**：BLEU / ROUGE-1/2/L / BERTScore（翻译和总结子集）
4. **检测并验证约束**：从指令中提取格式约束并验证输出是否符合（指令遵循子集）
5. **生成 Markdown 报告**：包含所有指标、约束类型分解、覆盖率统计

```bash
conda activate granite_ft

# 默认输出目录: evaluation/performance/{文件名}_eval/
python scripts/score.py \
  --input-file evaluation/output_data/Granite-v6_20260208_120000.json

# 自定义输出目录
python scripts/score.py \
  --input-file evaluation/output_data/Granite-v6_20260208_120000.json \
  --output-dir evaluation/performance/my_eval
```

**score.py 参数说明：**

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--input-file` | 是 | - | 推理结果文件（generate.py 的输出） |
| `--output-dir` | 否 | `evaluation/performance/{文件名}_eval/` | 评分结果输出目录 |

**输出位置：**
- `{output_dir}/eval_results.json` - JSON 格式评分结果
- `{output_dir}/eval_report.md` - Markdown 格式评估报告

**评分报告内容：**

| 任务类型 | 评估指标 | 说明 |
|----------|----------|------|
| 翻译 (translation) | BLEU, ROUGE-1/2/L, BERTScore | 需要 reference |
| 总结 (summarization) | BLEU, ROUGE-1/2/L, BERTScore | 需要 reference |
| 指令遵循 (instruction_following) | IFR, Strict Acc, Loose Acc, 约束类型分解 | 不依赖 reference，从指令提取约束并验证 |

**指令遵循评估详解：**

score.py 内置 35 种约束检测模式，覆盖中英文常见格式约束：

| 约束类别 | 示例 |
|----------|------|
| 长度约束 | "at least 200 words" / "不超过500字" / "exactly 3 paragraphs" |
| 格式约束 | "use bullet points" / "in json format" / "use markdown format" |
| 关键词约束 | "must include 'AI'" / "必须包含关键词：深度学习" / "do not use 'very'" |
| 语言约束 | "response should be in Chinese" |
| 结构约束 | "三段式结构" / "highlight at least 3 sections" / "end with a question" |
| IFEval标准约束 | postscript, quotation wrap, all uppercase/lowercase, repeat prompt |

**运行时重分类机制：**

score.py 在评分前会自动检查标记为 `instruction_following` 的样本：
- 有可检测约束 → 保持IF
- 来自公开基准（M-IFEval）且无reference → 保持IF
- 无约束 + 匹配总结/翻译模式 → 自动重分类，避免污染IF指标

##### 完整工作流示例

```bash
# 1. 批量推理多个模型
python scripts/generate.py \
  --models \
    "Base:D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16" \
    "Finetuned:D:/AI_code/models/AAITC-Quantum-Granite-4.0-v1.1.1-bf16:outputs/granite-4.0-1B-lora_v6_instruction" \
  --eval-file data/test_v4_enhanced.json

# 2. 分别评分
python scripts/score.py --input-file evaluation/output_data/Base_20260208_120000.json
python scripts/score.py --input-file evaluation/output_data/Finetuned_20260208_120100.json

# 3. 查看报告
# - evaluation/performance/Base_20260208_120000_eval/eval_report.md
# - evaluation/performance/Finetuned_20260208_120100_eval/eval_report.md
```

### 步骤4：导出合并模型（可选）

```bash
python run.py export \
  --adapter outputs/granite-4.0-1B-lora \
  --export-dir outputs/granite-4.0-1B-merged
```

## 评估指标说明

### 内容质量指标（翻译/总结）

| 指标 | 适用任务 | 说明 |
|------|----------|------|
| BLEU | 翻译 | n-gram精确度，业界标准翻译评估指标 |
| ROUGE-1/2/L | 总结/翻译 | n-gram召回率，摘要评估标准指标（中文自动jieba分词） |
| BERTScore | 通用 | 基于BERT的语义相似度，捕捉深层语义（自动检测中/英文选择模型） |

### 指令遵循指标

| 指标 | 说明 |
|------|------|
| IFR (Instruction Following Rate) | 约束通过率 = 通过的约束数 / 总约束数 |
| Strict Accuracy | 完全通过率 = 所有约束全部通过的样本占比 |
| Loose Accuracy | 宽松通过率 = 至少通过一半约束的样本占比 |
| 约束覆盖率 | 检测到约束的样本数 / 总IF样本数（越高越好） |

## 常见问题

**Q: 数据生成API报错？**
- 检查 `.env` 中的 API_KEY 是否正确（不要加引号）
- 确认 API_BASE_URL 格式正确（需以 `/v1` 结尾）
- 减小 `--workers` 避免触发限流

**Q: 如何使用自定义数据？**
- 按alpaca格式准备数据：`{"instruction": "...", "input": "...", "output": "..."}`
- 保存为JSON数组格式到 `data/train.json`
- 在 `data/dataset_info.json` 中注册

**Q: 为什么要聚焦单一领域？**
- 1B参数的小模型容量有限
- 跨领域+跨任务容易导致灾难性遗忘
- 锁定计算机/AI领域，模型只需学一套"行话"即可处理翻译和总结

## 技术栈

- LLaMA-Factory 0.9.x
- PyTorch 2.x + CUDA 12.1
- Transformers 4.45+
- PEFT (LoRA/QLoRA)
- 评估：sacrebleu, rouge-score, bert-score
