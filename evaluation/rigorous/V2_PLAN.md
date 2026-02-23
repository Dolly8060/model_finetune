# Qwen3-1.7B V2 微调完整执行方案

## 1. 目标定义
- 目标 1：保持翻译与总结能力不明显回退。
- 目标 2：提升指令遵循（IF）稳定性，重点提高 IFR。
- 目标 3：在 `labeled` 与 `if_unlabeled` 两套评测上都得到一致正向提升。

## 2. 成功标准（V2 验收线）
- 翻译：BLEU、ROUGE-L 不明显下降（建议下降不超过 1 分）。
- 总结：ROUGE-L、BERTScore-F1 不明显下降（建议下降不超过 1 分）。
- IF（labeled）：IFR 相比 V1 继续提升，Strict 不下降。
- IF（unlabeled）：IFR/Strict/Loose 同向提升。
- 报告口径：必须同时包含 `labeled + unlabeled` 两套结果。

## 3. 环境准备
```powershell
conda activate granite_ft
cd /d D:\AI_code\model_finetune
```

## 4. V2 实验矩阵（先做 3 组）
- `v2a`：IF 40%，翻译 30%，总结 30%。
- `v2b`：IF 50%，翻译 25%，总结 25%。
- `v2c`：与 `v2b` 同数据，但学习率更小（更稳）。

说明：
- 三组实验均使用同一验证集和同一测试集。
- 一次只改一个变量，便于判断因果关系。

## 5. 生成 V2 训练集（可直接复制）
```powershell
@'
import json, random
from pathlib import Path

random.seed(2026)
src = Path("data/qwen3_rigorous_train.json")
data = json.loads(src.read_text(encoding="utf-8"))

groups = {"translation": [], "summarization": [], "instruction_following": []}
for x in data:
    t = x.get("task_type")
    if t in groups:
        groups[t].append(x)

def take(pool, n):
    if n <= len(pool):
        return random.sample(pool, n)
    return pool + random.choices(pool, k=n-len(pool))

plans = {
    "v2a": {"translation": 1080, "summarization": 1080, "instruction_following": 1440},
    "v2b": {"translation": 900, "summarization": 900, "instruction_following": 1800},
    "v2c": {"translation": 900, "summarization": 900, "instruction_following": 1800},
}

for name, plan in plans.items():
    out = []
    for task, n in plan.items():
        out.extend(take(groups[task], n))
    random.shuffle(out)
    Path(f"data/qwen3_rigorous_train_{name}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8"
    )
    print(name, len(out), plan)
'@ | python -
```

## 6. 注册数据集（编辑 `data/dataset_info.json`）
在 JSON 顶层新增如下 3 个条目（注意逗号）：

```json
"qwen3_rigorous_train_v2a": {
  "file_name": "qwen3_rigorous_train_v2a.json",
  "formatting": "alpaca",
  "columns": { "prompt": "instruction", "query": "input", "response": "output" }
},
"qwen3_rigorous_train_v2b": {
  "file_name": "qwen3_rigorous_train_v2b.json",
  "formatting": "alpaca",
  "columns": { "prompt": "instruction", "query": "input", "response": "output" }
},
"qwen3_rigorous_train_v2c": {
  "file_name": "qwen3_rigorous_train_v2c.json",
  "formatting": "alpaca",
  "columns": { "prompt": "instruction", "query": "input", "response": "output" }
}
```

## 7. 复制配置并创建 3 个训练 yaml
```powershell
Copy-Item configs/finetune_qwen3_lora_rigorous.yaml configs/finetune_qwen3_lora_v2a.yaml
Copy-Item configs/finetune_qwen3_lora_rigorous.yaml configs/finetune_qwen3_lora_v2b.yaml
Copy-Item configs/finetune_qwen3_lora_rigorous.yaml configs/finetune_qwen3_lora_v2c.yaml
```

修改项：
- `configs/finetune_qwen3_lora_v2a.yaml`
  - `dataset: qwen3_rigorous_train_v2a`
  - `output_dir: ./outputs/qwen3-1.7B-lora-v2a`
- `configs/finetune_qwen3_lora_v2b.yaml`
  - `dataset: qwen3_rigorous_train_v2b`
  - `output_dir: ./outputs/qwen3-1.7B-lora-v2b`
- `configs/finetune_qwen3_lora_v2c.yaml`
  - `dataset: qwen3_rigorous_train_v2c`
  - `output_dir: ./outputs/qwen3-1.7B-lora-v2c`
  - `learning_rate: 6.0e-5`

## 8. 训练命令
```powershell
llamafactory-cli train configs/finetune_qwen3_lora_v2a.yaml
llamafactory-cli train configs/finetune_qwen3_lora_v2b.yaml
llamafactory-cli train configs/finetune_qwen3_lora_v2c.yaml
```

## 9. 评测流程（每个实验都跑 labeled + unlabeled）
以下以 `v2a` 为例，`v2b`/`v2c` 按名称替换。

### 9.1 labeled（有标注）
```powershell
python scripts/generate.py --models "FTQwen3_V2A:D:/AI_code/models/Qwen3-1.7B:outputs/qwen3-1.7B-lora-v2a" --eval-file data/qwen3_rigorous_test_labeled.json --output-file evaluation/output_data/v2a_labeled.json --max-input-length 2048
python scripts/score.py --input-file evaluation/output_data/v2a_labeled.json --output-dir evaluation/rigorous/v2a_labeled
```

### 9.2 if_unlabeled（无标注 IF）
```powershell
python scripts/generate.py --models "FTQwen3_V2A:D:/AI_code/models/Qwen3-1.7B:outputs/qwen3-1.7B-lora-v2a" --eval-file data/qwen3_rigorous_test_if_unlabeled.json --output-file evaluation/output_data/v2a_if_unlabeled.json --max-input-length 2048
python scripts/score.py --input-file evaluation/output_data/v2a_if_unlabeled.json --output-dir evaluation/rigorous/v2a_if_unlabeled
```

## 10. 对比判定方法（如何选最优 V2）
先看 `labeled`：
- 翻译/总结不回退。
- IF 的 IFR 与 Strict 尽量同时提升。

再看 `unlabeled`：
- IFR/Strict/Loose 是否同向提升。
- 重点观察格式约束是否改善（如 `json_format`、`bullet_points`、`placeholder_count`）。

最终推荐规则：
- 若 `v2b` IF 提升最大但翻译总结回退明显，优先选择更稳的 `v2c`。
- 若 `v2a` 在综合指标最平衡，优先选择 `v2a`。

## 11. 结果归档规范
每个实验至少保留：
- 训练输出目录：`outputs/qwen3-1.7B-lora-v2*`
- 推理结果：`evaluation/output_data/v2*_labeled.json`、`evaluation/output_data/v2*_if_unlabeled.json`
- 评分结果：`evaluation/rigorous/v2*_labeled/eval_results.json`
- 评分结果：`evaluation/rigorous/v2*_if_unlabeled/eval_results.json`

## 12. 常见问题
- `ModuleNotFoundError`：说明当前未激活 `granite_ft` 或缺依赖，先 `conda activate granite_ft`。
- 显存不足：降低 `per_device_train_batch_size`，必要时将 `cutoff_len` 调到 1536。
- 推理过慢：先用 `--max-samples 100` 做烟雾测试再全量跑。

## 13. 下一步（执行顺序）
1. 先生成 `v2a/v2b/v2c` 数据集并注册 `dataset_info.json`。
2. 先跑 `v2a` 训练 + 双评测，确认全流程通。
3. 再跑 `v2b`、`v2c`，做三组横向对比。
4. 生成 `conclusion_v3.md`，形成最终选型结论。
