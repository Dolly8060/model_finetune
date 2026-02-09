#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量推理脚本 - 在测试集上运行多个模型并保存预测结果

功能：
1. 支持多个模型批量推理
2. 自动检测模型类型并使用对应的prompt模板
3. 保存详细的预测结果到 data/output_data/

输出格式：
    data/output_data/{model_name}_{timestamp}.json
"""

import os
import json
import argparse
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 模型模板配置
# ============================================================

MODEL_TEMPLATES = {
    "granite": {
        "format": "<|start_of_role|>user<|end_of_role|>{instruction}\n\n{input}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
    },
    "qwen": {
        "format": "<|im_start|>user\n{instruction}\n\n{input}<|im_end|>\n<|im_start|>assistant\n",
    },
    "llama": {
        "format": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "chatml": {
        "format": "<|im_start|>user\n{instruction}\n\n{input}<|im_end|>\n<|im_start|>assistant\n",
    },
    "default": {
        "format": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    },
}

# 任务类型识别关键词
TRANSLATION_KEYWORDS = [
    "翻译", "translate", "translation", "译为", "译成",
    "english translation", "中文翻译", "中译英", "英译中",
]

SUMMARY_KEYWORDS = [
    "总结", "摘要", "概括", "summary", "summarize",
    "提炼", "提取主要观点", "总结以下", "概括下文",
]


# ============================================================
# 工具函数
# ============================================================

def detect_model_type(model_name: str) -> str:
    """根据模型名称检测模型类型"""
    model_name_lower = model_name.lower()
    if "granite" in model_name_lower:
        return "granite"
    elif "qwen" in model_name_lower:
        return "qwen"
    elif "llama" in model_name_lower:
        return "llama"
    elif "gemma" in model_name_lower:
        return "chatml"
    else:
        return "default"


def format_prompt(instruction: str, input_text: str, model_type: str) -> str:
    """根据模型类型格式化prompt"""
    template = MODEL_TEMPLATES.get(model_type, MODEL_TEMPLATES["default"])
    return template["format"].format(instruction=instruction, input=input_text)


def classify_sample(sample: Dict) -> Optional[str]:
    """根据instruction或task_type判断任务类型"""
    task_type = sample.get("task_type", "").lower()
    if task_type in ("translation", "summarization", "instruction_following"):
        return task_type
    
    instr = sample.get("instruction", "").lower()
    if any(kw in instr for kw in TRANSLATION_KEYWORDS):
        return "translation"
    if any(kw in instr for kw in SUMMARY_KEYWORDS):
        return "summarization"
    
    # 默认为指令遵循
    return "instruction_following"


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: str = None,
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """加载模型和tokenizer"""
    print(f"加载模型: {model_path}")
    
    model_type = detect_model_type(model_path)
    print(f"  检测到模型类型: {model_type}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    if adapter_path:
        print(f"  加载LoRA适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer, model_type


def release_model(model, tokenizer):
    """释放模型显存"""
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("  已释放显存")


def generate_response(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 512
) -> str:
    """生成模型回复"""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1536
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()


# ============================================================
# 主要推理逻辑
# ============================================================

def run_inference(
    model,
    tokenizer,
    model_type: str,
    eval_data: List[Dict],
    model_name: str,
    max_new_tokens: int = 512,
) -> List[Dict]:
    """对评估数据运行推理"""
    results = []
    
    for sample in tqdm(eval_data, desc=f"推理 {model_name}"):
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        reference = sample.get("output", "")
        task_type = classify_sample(sample)
        
        # 格式化prompt
        prompt = format_prompt(instruction, input_text, model_type)
        
        # 生成预测
        try:
            prediction = generate_response(model, tokenizer, prompt, max_new_tokens)
        except Exception as e:
            print(f"  推理失败: {e}")
            prediction = f"[ERROR] {str(e)}"
        
        results.append({
            "instruction": instruction,
            "input": input_text[:500] + "..." if len(input_text) > 500 else input_text,
            "reference": reference,
            "prediction": prediction,
            "task_type": task_type,
        })
    
    return results


def save_results(
    results: List[Dict],
    model_name: str,
    model_path: str,
    adapter_path: Optional[str],
    eval_file: str,
    output_dir: str,
    output_file: Optional[str] = None,
) -> str:
    """保存推理结果
    
    Args:
        output_dir: 输出目录
        output_file: 自定义输出文件路径（如果指定则忽略 output_dir）
    """
    # 确定输出文件路径
    if output_file:
        # 使用自定义路径
        final_output_file = output_file
        os.makedirs(os.path.dirname(final_output_file) or ".", exist_ok=True)
    else:
        # 使用默认格式: output_dir/{model_name}_{timestamp}.json
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^\w\-]', '_', model_name)
        final_output_file = os.path.join(output_dir, f"{safe_name}_{timestamp}.json")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_data = {
        "model_name": model_name,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "eval_file": eval_file,
        "timestamp": timestamp,
        "total_samples": len(results),
        "samples": results,
    }
    
    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存: {final_output_file}")
    return final_output_file


def parse_model_spec(spec: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    解析模型配置字符串
    格式: 名称:模型路径 或 名称:模型路径:adapter路径
    """
    pattern = r'^([^:]+):([A-Za-z]:)?([^:]+)(?::([A-Za-z]:)?([^:]+))?$'
    match = re.match(pattern, spec)
    
    if match:
        name = match.group(1)
        drive1 = match.group(2) or ""
        path1 = match.group(3)
        drive2 = match.group(4) or ""
        path2 = match.group(5)
        
        model_path = drive1 + path1
        adapter_path = (drive2 + path2) if path2 else None
        
        return name, model_path, adapter_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="批量推理脚本 - 在测试集上运行多个模型并保存预测结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  1. 单模型推理（默认输出路径）:
     python generate.py --models "Granite-1B:D:/models/granite-1b" --eval-file data/test_v4.json

  2. 单模型推理（自定义输出路径）:
     python generate.py --models "Granite-1B:D:/models/granite-1b" --eval-file data/test.json --output-file evaluation/my_results.json

  3. 多模型推理:
     python generate.py --models \\
       "Base-Granite:D:/models/granite-1b" \\
       "Finetuned-Granite:D:/models/granite-1b:outputs/lora_v6" \\
       "Qwen3-4B:D:/models/qwen3-4b"

  4. 限制样本数:
     python generate.py --models "Model:path" --eval-file data/test.json --max-samples 50
        """
    )
    
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="模型配置列表，格式: '名称:模型路径' 或 '名称:模型路径:adapter路径'"
    )
    parser.add_argument(
        "--eval-file", type=str, required=True,
        help="评估数据文件路径"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation/output_data",
        help="输出目录 (默认: data/output_data)"
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="自定义输出文件路径 (仅单模型时有效，覆盖 --output-dir)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="最大评估样本数，0表示全部 (默认: 0)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="生成的最大token数 (默认: 512)"
    )
    
    args = parser.parse_args()
    
    # 解析模型配置
    model_configs = []
    for spec in args.models:
        result = parse_model_spec(spec)
        if result is None:
            print(f"错误: 无效的模型配置格式: {spec}")
            print("正确格式: '名称:模型路径' 或 '名称:模型路径:adapter路径'")
            return
        name, path, adapter = result
        model_configs.append({"name": name, "path": path, "adapter": adapter})
    
    # 加载评估数据
    print(f"\n加载评估数据: {args.eval_file}")
    with open(args.eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    
    if args.max_samples > 0 and len(eval_data) > args.max_samples:
        import random
        random.seed(42)
        eval_data = random.sample(eval_data, args.max_samples)
    
    print(f"评估样本数: {len(eval_data)}")
    
    # 显示模型信息
    print(f"\n将对 {len(model_configs)} 个模型进行推理:")
    for config in model_configs:
        adapter_info = f" (adapter: {config['adapter']})" if config['adapter'] else ""
        print(f"  - {config['name']}: {config['path']}{adapter_info}")
    
    # 检查 --output-file 参数
    if args.output_file and len(model_configs) > 1:
        print(f"\n警告: --output-file 参数仅对单模型有效，多模型推理时将使用 --output-dir")
        args.output_file = None
    
    # 逐个模型推理
    output_files = []
    for i, config in enumerate(model_configs):
        model_name = config["name"]
        model_path = config["path"]
        adapter_path = config["adapter"]
        
        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{len(model_configs)}] 推理模型: {model_name}")
        print(f"{'=' * 60}")
        
        try:
            # 加载模型
            model, tokenizer, model_type = load_model_and_tokenizer(
                model_path, adapter_path
            )
            
            # 运行推理
            results = run_inference(
                model, tokenizer, model_type, eval_data, 
                model_name, args.max_new_tokens
            )
            
            # 保存结果
            output_file = save_results(
                results, model_name, model_path, adapter_path,
                args.eval_file, args.output_dir, args.output_file
            )
            output_files.append(output_file)
            
            # 释放显存
            release_model(model, tokenizer)
            
        except Exception as e:
            import traceback
            print(f"\n{'!'*60}")
            print(f"!!! 模型 {model_name} 推理失败 !!!")
            print(f"!!! 错误: {e}")
            print(f"{'!'*60}")
            traceback.print_exc()
            
            # 尝试清理显存
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass
    
    # 汇总
    print(f"\n{'=' * 60}")
    print("推理完成！")
    print(f"{'=' * 60}")
    print(f"成功生成 {len(output_files)} 个结果文件:")
    for f in output_files:
        print(f"  - {f}")
    print(f"\n下一步: 使用 score.py 对结果进行评分")
    print(f"示例: python scripts/score.py --input-file {output_files[0] if output_files else 'data/output_data/xxx.json'}")


if __name__ == "__main__":
    main()
