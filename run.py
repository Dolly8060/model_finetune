#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Granite 4.0-1B 微调主脚本
使用LLaMA-Factory进行LoRA/QLoRA微调
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def run_command(cmd: list, env: dict = None):
    """执行命令"""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=full_env)
    return result.returncode


def generate_dataset(args):
    """生成数据集"""
    cmd = [
        sys.executable, "scripts/generate_dataset.py",
        "--total", str(args.total),
        "--translation-ratio", str(args.translation_ratio),
        "--output", args.output,
        "--workers", str(args.workers),
    ]
    return run_command(cmd)


def train_model(args):
    """训练模型"""
    config_path = args.config
    
    # 替换配置中的环境变量
    model_path = os.getenv("MODEL_PATH", args.model_path)
    if not model_path:
        print("错误: 请设置MODEL_PATH环境变量或使用--model-path参数")
        return 1
    
    # 使用llamafactory-cli进行训练
    cmd = [
        "llamafactory-cli", "train", config_path
    ]
    
    # 设置环境变量
    env = {"MODEL_PATH": model_path}
    
    return run_command(cmd, env)


def evaluate_model(args):
    """评估模型"""
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--base-model", args.base_model,
        "--eval-file", args.eval_file,
        "--max-samples", str(args.max_samples),
        "--output-dir", args.output_dir,
    ]
    
    if args.finetuned_model:
        cmd.extend(["--finetuned-model", args.finetuned_model])
    if args.adapter:
        cmd.extend(["--adapter", args.adapter])
    
    return run_command(cmd)


def export_model(args):
    """导出合并后的模型"""
    model_path = os.getenv("MODEL_PATH", args.model_path)
    
    cmd = [
        "llamafactory-cli", "export",
        "--model_name_or_path", model_path,
        "--adapter_name_or_path", args.adapter,
        "--template", "granite4",
        "--finetuning_type", "lora",
        "--export_dir", args.export_dir,
        "--export_size", "2",
        "--export_legacy_format", "false",
    ]
    
    return run_command(cmd)


def main():
    parser = argparse.ArgumentParser(description="Granite模型微调工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 生成数据集
    gen_parser = subparsers.add_parser("generate", help="生成训练数据集")
    gen_parser.add_argument("--total", type=int, default=2000, help="总样本数")
    gen_parser.add_argument("--translation-ratio", type=float, default=0.5, help="翻译数据占比")
    gen_parser.add_argument("--output", type=str, default="data/train.json", help="输出文件")
    gen_parser.add_argument("--workers", type=int, default=5, help="并行线程数")
    
    # 训练模型
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--config", type=str, default="configs/finetune_lora.yaml",
        help="训练配置文件"
    )
    train_parser.add_argument("--model-path", type=str, help="模型路径")
    
    # 评估模型
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("--base-model", type=str, required=True, help="基础模型路径")
    eval_parser.add_argument("--finetuned-model", type=str, help="微调模型路径")
    eval_parser.add_argument("--adapter", type=str, help="LoRA adapter路径")
    eval_parser.add_argument("--eval-file", type=str, default="data/val.json", help="评估数据")
    eval_parser.add_argument("--max-samples", type=int, default=100, help="最大评估样本数")
    eval_parser.add_argument("--output-dir", type=str, default="evaluation", help="输出目录")
    
    # 导出模型
    export_parser = subparsers.add_parser("export", help="导出合并后的模型")
    export_parser.add_argument("--model-path", type=str, help="基础模型路径")
    export_parser.add_argument("--adapter", type=str, required=True, help="LoRA adapter路径")
    export_parser.add_argument("--export-dir", type=str, required=True, help="导出目录")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 切换到脚本所在目录
    os.chdir(Path(__file__).parent)
    
    if args.command == "generate":
        return generate_dataset(args)
    elif args.command == "train":
        return train_model(args)
    elif args.command == "evaluate":
        return evaluate_model(args)
    elif args.command == "export":
        return export_model(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
