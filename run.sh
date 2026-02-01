#!/bin/bash
# Granite 4.0-1B 微调工具 - Linux/Mac启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活conda环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate granite_finetune 2>/dev/null

if [ $? -ne 0 ]; then
    echo "[提示] 未找到granite_finetune环境，请先运行: conda env create -f environment.yml"
    exit 1
fi

# 执行命令
python run.py "$@"
