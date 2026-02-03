#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IFEval / M-IFEval 数据集下载与转换脚本

功能：
1. 下载 Google IFEval 指令遵循评估数据集 (~541条)
2. 下载 argilla/ifeval-like-data 带响应的指令数据 (100K+)
3. 下载 facebook/Multi-IF 多语言指令遵循数据
4. 转换为 Alpaca 格式
5. 支持 API 生成中文指令遵循数据补充

数据集来源：
- IFEval: huggingface.co/datasets/google/IFEval (原始评估集，无响应)
- argilla/ifeval-like-data: 带响应的合成数据 (推荐训练用)
- facebook/Multi-IF: 多语言指令遵循数据

输出格式：
{
    "instruction": "Write exactly 3 paragraphs...",
    "input": "",
    "output": "[生成的符合指令的响应]",
    "task_type": "instruction_following"
}
"""

import os
import sys
import json
import argparse
import random
import time
import re
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from datasets import load_dataset
except ImportError:
    print("请先安装datasets库: pip install datasets")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("警告: 未安装openai库，将无法生成响应。安装命令: pip install openai")

# 随机种子
random.seed(2026)


# ============================================================
# 中文指令遵循模板 (用于API生成补充数据)
# ============================================================

# 指令类型及对应的中文模板
CHINESE_INSTRUCTION_TEMPLATES = {
    "word_count": [
        "请用不少于{min_words}个字描述{topic}。",
        "写一段关于{topic}的文字，字数要求在{min_words}到{max_words}之间。",
        "用恰好{exact_words}个字概括{topic}的核心要点。",
    ],
    "format_constraint": [
        "请用列表形式列出{topic}的{count}个要点。",
        "以问答形式介绍{topic}，包含至少{count}个问题和回答。",
        "用Markdown格式撰写关于{topic}的说明，必须包含标题和分点。",
        "请用JSON格式输出{topic}的关键信息。",
    ],
    "keyword_inclusion": [
        "写一段关于{topic}的文字，必须包含以下关键词：{keywords}。",
        "在描述{topic}时，请确保'{keyword}'这个词出现至少{count}次。",
        "撰写{topic}的介绍，每段必须包含'{keyword}'。",
    ],
    "structure_constraint": [
        "请写{count}段关于{topic}的内容，每段以数字编号开头。",
        "用三段式结构（引言、正文、结论）介绍{topic}。",
        "写一篇关于{topic}的短文，必须以问句结尾。",
        "描述{topic}，开头必须是'{start_phrase}'。",
    ],
    "exclusion_constraint": [
        "介绍{topic}，但不要使用'{forbidden_word}'这个词。",
        "写一段关于{topic}的文字，不要使用任何形容词。",
        "描述{topic}的特点，不要使用'非常'、'很'等程度副词。",
    ],
    "language_style": [
        "用正式的学术语言描述{topic}。",
        "用通俗易懂的语言向初学者解释{topic}。",
        "以幽默风趣的方式介绍{topic}。",
        "用简洁的技术文档风格说明{topic}。",
    ],
}

# 话题列表（计算机/AI领域）
CS_AI_TOPICS = [
    "深度学习", "神经网络", "自然语言处理", "计算机视觉", "机器学习",
    "Transformer架构", "注意力机制", "大语言模型", "模型压缩", "知识蒸馏",
    "强化学习", "生成对抗网络", "自监督学习", "迁移学习", "联邦学习",
    "边缘计算", "模型量化", "分布式训练", "提示工程", "RAG检索增强",
    "代码生成", "文本分类", "情感分析", "命名实体识别", "机器翻译",
    "语音识别", "图像分割", "目标检测", "推荐系统", "异常检测",
]

# 关键词列表
KEYWORDS = [
    "人工智能", "深度学习", "神经网络", "算法", "模型", "训练", "推理",
    "参数", "优化", "损失函数", "梯度下降", "反向传播", "特征提取",
]


def generate_chinese_if_instructions(count: int = 500) -> List[Dict]:
    """生成中文指令遵循数据模板
    
    Args:
        count: 生成数量
        
    Returns:
        指令模板列表，包含 instruction 和验证规则
    """
    instructions = []
    
    for _ in range(count):
        # 随机选择指令类型
        instr_type = random.choice(list(CHINESE_INSTRUCTION_TEMPLATES.keys()))
        template = random.choice(CHINESE_INSTRUCTION_TEMPLATES[instr_type])
        topic = random.choice(CS_AI_TOPICS)
        
        # 根据类型填充参数
        params = {"topic": topic}
        verifiable_rules = []
        
        if instr_type == "word_count":
            if "{min_words}" in template:
                params["min_words"] = random.choice([100, 150, 200, 250, 300])
                verifiable_rules.append(("min_length", params["min_words"]))
            if "{max_words}" in template:
                params["max_words"] = params.get("min_words", 100) + random.choice([100, 150, 200])
                verifiable_rules.append(("max_length", params["max_words"]))
            if "{exact_words}" in template:
                params["exact_words"] = random.choice([50, 80, 100, 120, 150])
                verifiable_rules.append(("exact_length", params["exact_words"]))
                
        elif instr_type == "format_constraint":
            params["count"] = random.choice([3, 4, 5, 6])
            # 安全提取格式类型
            format_type = "list"
            if "形式" in template and "用" in template:
                try:
                    format_type = template.split("用")[1].split("形式")[0]
                except (IndexError, ValueError):
                    format_type = "list"
            verifiable_rules.append(("format", format_type))
            
        elif instr_type == "keyword_inclusion":
            keyword = random.choice(KEYWORDS)
            params["keyword"] = keyword
            params["keywords"] = ", ".join(random.sample(KEYWORDS, 3))
            params["count"] = random.choice([2, 3, 4])
            verifiable_rules.append(("keyword_count", (keyword, params.get("count", 1))))
            
        elif instr_type == "structure_constraint":
            params["count"] = random.choice([3, 4, 5])
            params["start_phrase"] = random.choice(["在当今时代", "众所周知", "随着技术发展"])
            verifiable_rules.append(("structure", instr_type))
            
        elif instr_type == "exclusion_constraint":
            params["forbidden_word"] = random.choice(["非常", "很", "特别", "极其"])
            verifiable_rules.append(("exclude_word", params["forbidden_word"]))
            
        elif instr_type == "language_style":
            verifiable_rules.append(("style", instr_type))
        
        try:
            instruction = template.format(**params)
        except KeyError:
            continue
            
        instructions.append({
            "instruction": instruction,
            "input": "",
            "instruction_type": instr_type,
            "verifiable_rules": verifiable_rules,
            "topic": topic,
        })
    
    return instructions


def download_ifeval_dataset() -> List[Dict]:
    """下载 Google IFEval 数据集（原始评估集，无响应）
    
    Returns:
        IFEval 数据列表（仅用于测试评估，无output）
    """
    print("下载 Google IFEval 原始数据集（用于评估）...")
    
    try:
        dataset = load_dataset("google/IFEval", split="train")
        print(f"  成功加载 {len(dataset)} 条 IFEval 数据")
    except Exception as e:
        print(f"  下载 IFEval 失败: {e}")
        return []
    
    eval_data = []
    for item in tqdm(dataset, desc="处理IFEval数据"):
        prompt = item.get("prompt", "")
        instruction_id_list = item.get("instruction_id_list", [])
        kwargs = item.get("kwargs", [])
        
        if not prompt:
            continue
        
        eval_data.append({
            "instruction": prompt,
            "input": "",
            "output": "",  # 原始数据无响应，仅用于评估
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs,
            "task_type": "instruction_following",
            "source": "ifeval_eval",
            "language": "en",
        })
    
    print(f"  成功处理 {len(eval_data)} 条 IFEval 评估数据")
    return eval_data


def download_argilla_ifeval_like(max_samples: int = 2000) -> List[Dict]:
    """下载 argilla/ifeval-like-data 带响应的指令遵循数据
    
    这是一个高质量的合成数据集，包含完整的 instruction + response
    
    Args:
        max_samples: 最大样本数
        
    Returns:
        带响应的指令遵循数据列表
    """
    print(f"下载 argilla/ifeval-like-data 数据集（最多 {max_samples} 条）...")
    
    try:
        if max_samples > 0:
            dataset = load_dataset("argilla/ifeval-like-data", split=f"train[:{max_samples}]")
        else:
            dataset = load_dataset("argilla/ifeval-like-data", split="train")
        print(f"  成功加载 {len(dataset)} 条数据")
    except Exception as e:
        print(f"  下载 argilla/ifeval-like-data 失败: {e}")
        return []
    
    eval_data = []
    for item in tqdm(dataset, desc="处理argilla数据"):
        instruction = item.get("instruction", "")
        response = item.get("response", "")
        
        if not instruction or not response:
            continue
        
        eval_data.append({
            "instruction": instruction,
            "input": "",
            "output": response,
            "task_type": "instruction_following",
            "source": "argilla_ifeval",
            "language": "en",
        })
    
    print(f"  成功处理 {len(eval_data)} 条带响应的指令遵循数据")
    return eval_data


def download_multi_if(max_samples: int = 1000) -> List[Dict]:
    """下载 facebook/Multi-IF 多语言指令遵循数据
    
    包含英语、法语、西班牙语等多语言数据
    
    Args:
        max_samples: 最大样本数
        
    Returns:
        多语言指令遵循数据列表
    """
    print(f"下载 facebook/Multi-IF 多语言数据集（最多 {max_samples} 条）...")
    
    try:
        if max_samples > 0:
            dataset = load_dataset("facebook/Multi-IF", split=f"train[:{max_samples}]")
        else:
            dataset = load_dataset("facebook/Multi-IF", split="train")
        print(f"  成功加载 {len(dataset)} 条数据")
    except Exception as e:
        print(f"  下载 facebook/Multi-IF 失败: {e}")
        return []
    
    eval_data = []
    for item in tqdm(dataset, desc="处理Multi-IF数据"):
        # Multi-IF 数据结构是多轮对话，提取第一轮
        turn_1_prompt = item.get("turn_1_prompt", {})
        language = item.get("language", "en")
        
        if isinstance(turn_1_prompt, dict):
            instruction = turn_1_prompt.get("content", "")
        elif isinstance(turn_1_prompt, str):
            instruction = turn_1_prompt
        else:
            continue
        
        if not instruction:
            continue
        
        # Multi-IF 本身也没有标准响应，但可以用于评估
        eval_data.append({
            "instruction": instruction,
            "input": "",
            "output": "",  # 需要生成或用于评估
            "task_type": "instruction_following",
            "source": "multi_if",
            "language": language.lower() if language else "en",
        })
    
    print(f"  成功处理 {len(eval_data)} 条 Multi-IF 数据")
    return eval_data


def download_mifeval_dataset() -> List[Dict]:
    """下载 M-IFEval 多语言数据集（已弃用，使用 Multi-IF 替代）
    
    Returns:
        空列表（该数据集不可用）
    """
    print("M-IFEval 官方数据集不可用，请使用 facebook/Multi-IF 替代")
    return []


def call_api_for_response(
    client: "OpenAI",
    instruction: str,
    model: str = "deepseek-chat",
    max_retries: int = 3
) -> Optional[str]:
    """调用 API 生成符合指令的响应
    
    Args:
        client: OpenAI 客户端
        instruction: 指令内容
        model: 模型名称
        max_retries: 最大重试次数
        
    Returns:
        生成的响应文本
    """
    system_prompt = """You are a helpful assistant that follows instructions precisely.
Your task is to generate a response that strictly follows the given instruction.
Pay close attention to any constraints mentioned in the instruction, such as:
- Word count limits (minimum, maximum, or exact)
- Format requirements (lists, paragraphs, JSON, etc.)
- Required keywords or phrases
- Forbidden words or phrases
- Structural requirements (number of paragraphs, ending with a question, etc.)

Generate a response that satisfies ALL constraints in the instruction."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction}
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"  API 调用失败: {e}")
                return None
    return None


def generate_responses_with_api(
    data: List[Dict],
    max_samples: int = 0,
    workers: int = 5
) -> List[Dict]:
    """使用 API 为数据生成响应
    
    Args:
        data: 输入数据列表
        max_samples: 最大处理样本数，0表示全部
        workers: 并行线程数
        
    Returns:
        带响应的数据列表
    """
    if not HAS_OPENAI:
        print("警告: 未安装 openai 库，无法生成响应")
        return data
    
    api_base = os.getenv("API_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.getenv("API_KEY", "")
    api_model = os.getenv("API_MODEL", "deepseek-chat")
    
    if not api_key:
        print("警告: 未配置 API_KEY，无法生成响应")
        print("请在 .env 文件中配置 API_KEY")
        return data
    
    client = OpenAI(base_url=api_base, api_key=api_key)
    
    # 过滤需要生成响应的数据
    to_generate = [d for d in data if not d.get("output")]
    if max_samples > 0:
        to_generate = to_generate[:max_samples]
    
    print(f"\n使用 API 生成 {len(to_generate)} 条响应...")
    print(f"  API: {api_base}")
    print(f"  Model: {api_model}")
    
    results = []
    
    def process_item(item):
        instruction = item["instruction"]
        if item.get("input"):
            instruction = f"{instruction}\n\n{item['input']}"
        
        response = call_api_for_response(client, instruction, api_model)
        if response:
            item["output"] = response
        return item
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_item, item): item for item in to_generate}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="生成响应"):
            result = future.result()
            if result.get("output"):
                results.append(result)
    
    # 合并已有响应的数据
    existing_with_output = [d for d in data if d.get("output") and d not in to_generate]
    results.extend(existing_with_output)
    
    print(f"  成功生成 {len(results)} 条带响应的数据")
    return results


def convert_to_alpaca_format(data: List[Dict]) -> List[Dict]:
    """转换为 Alpaca 格式
    
    Args:
        data: 原始数据列表
        
    Returns:
        Alpaca 格式数据列表
    """
    alpaca_data = []
    for item in data:
        if not item.get("output"):
            continue
        
        alpaca_data.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"],
        })
    
    return alpaca_data


def main():
    parser = argparse.ArgumentParser(description="下载 IFEval 指令遵循数据集（混合来源）")
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="输出目录"
    )
    parser.add_argument(
        "--argilla-samples", type=int, default=1500,
        help="从 argilla/ifeval-like-data 下载的样本数（带响应）"
    )
    parser.add_argument(
        "--multi-if-samples", type=int, default=500,
        help="从 facebook/Multi-IF 下载的样本数"
    )
    parser.add_argument(
        "--chinese-samples", type=int, default=600,
        help="生成的中文指令样本数"
    )
    parser.add_argument(
        "--generate-responses", action="store_true",
        help="使用 API 为无响应数据生成响应"
    )
    parser.add_argument(
        "--max-api-samples", type=int, default=600,
        help="API 生成的最大样本数"
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="API 并行请求数"
    )
    parser.add_argument(
        "--skip-argilla", action="store_true",
        help="跳过 argilla/ifeval-like-data 下载"
    )
    parser.add_argument(
        "--skip-multi-if", action="store_true",
        help="跳过 Multi-IF 下载"
    )
    parser.add_argument(
        "--skip-ifeval-eval", action="store_true",
        help="跳过 IFEval 评估集下载"
    )
    parser.add_argument(
        "--skip-chinese", action="store_true",
        help="跳过中文数据生成"
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_data = []
    eval_data = []  # 仅用于评估的数据（无响应）
    
    # 1. 下载 argilla/ifeval-like-data（主要训练数据，带响应）
    if not args.skip_argilla:
        argilla_data = download_argilla_ifeval_like(max_samples=args.argilla_samples)
        all_data.extend(argilla_data)
        
        if argilla_data:
            argilla_path = os.path.join(args.output_dir, "argilla_ifeval.json")
            with open(argilla_path, "w", encoding="utf-8") as f:
                json.dump(argilla_data, f, ensure_ascii=False, indent=2)
            print(f"  已保存: {argilla_path}")
    
    # 2. 下载 facebook/Multi-IF（多语言数据）
    if not args.skip_multi_if:
        multi_if_data = download_multi_if(max_samples=args.multi_if_samples)
        all_data.extend(multi_if_data)
        
        if multi_if_data:
            multi_if_path = os.path.join(args.output_dir, "multi_if.json")
            with open(multi_if_path, "w", encoding="utf-8") as f:
                json.dump(multi_if_data, f, ensure_ascii=False, indent=2)
            print(f"  已保存: {multi_if_path}")
    
    # 3. 下载 IFEval 原始评估集（仅用于测试评估，不用于训练）
    if not args.skip_ifeval_eval:
        ifeval_eval_data = download_ifeval_dataset()
        eval_data.extend(ifeval_eval_data)
        
        if ifeval_eval_data:
            ifeval_raw_path = os.path.join(args.output_dir, "ifeval_eval.json")
            with open(ifeval_raw_path, "w", encoding="utf-8") as f:
                json.dump(ifeval_eval_data, f, ensure_ascii=False, indent=2)
            print(f"  已保存: {ifeval_raw_path}（评估用）")
    
    # 4. 生成中文指令遵循数据
    if not args.skip_chinese and args.chinese_samples > 0:
        print(f"\n生成 {args.chinese_samples} 条中文指令遵循模板...")
        chinese_data = generate_chinese_if_instructions(args.chinese_samples)
        
        # 添加元信息
        for item in chinese_data:
            item["task_type"] = "instruction_following"
            item["source"] = "chinese_generated"
            item["language"] = "zh"
            item["output"] = ""  # 需要生成
        
        all_data.extend(chinese_data)
        
        # 保存中文模板
        chinese_template_path = os.path.join(args.output_dir, "ifeval_chinese_templates.json")
        with open(chinese_template_path, "w", encoding="utf-8") as f:
            json.dump(chinese_data, f, ensure_ascii=False, indent=2)
        print(f"  已保存: {chinese_template_path}")
    
    # 5. 使用 API 为无响应数据生成响应
    if args.generate_responses:
        # 筛选需要生成响应的数据
        need_response = [d for d in all_data if not d.get("output")]
        has_response = [d for d in all_data if d.get("output")]
        
        print(f"\n已有响应: {len(has_response)} 条")
        print(f"需要生成响应: {len(need_response)} 条")
        
        if need_response:
            # 优先为中文数据生成响应（因为英文已有argilla数据）
            chinese_need = [d for d in need_response if d.get("language") in ["zh", "chinese"]]
            other_need = [d for d in need_response if d.get("language") not in ["zh", "chinese"]]
            
            print(f"  其中中文: {len(chinese_need)} 条（优先生成）")
            print(f"  其中其他语言: {len(other_need)} 条")
            
            # 优先生成中文，剩余配额给其他语言
            chinese_quota = min(args.max_api_samples, len(chinese_need))
            other_quota = max(0, args.max_api_samples - chinese_quota)
            
            to_generate = chinese_need[:chinese_quota] + other_need[:other_quota]
            
            generated_data = generate_responses_with_api(
                to_generate,
                max_samples=len(to_generate),
                workers=args.workers
            )
            # 合并已有响应和新生成的响应
            all_data = has_response + generated_data
    
    # 6. 转换为 Alpaca 格式并保存
    alpaca_data = convert_to_alpaca_format(all_data)
    
    if alpaca_data:
        # 保存完整数据（带元信息）
        full_path = os.path.join(args.output_dir, "ifeval_full_with_meta.json")
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump([d for d in all_data if d.get("output")], f, ensure_ascii=False, indent=2)
        
        # 保存 Alpaca 格式数据（用于训练）
        alpaca_path = os.path.join(args.output_dir, "ifeval_alpaca.json")
        with open(alpaca_path, "w", encoding="utf-8") as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n已保存 Alpaca 格式数据: {alpaca_path}")
    
    # 7. 保存评估集（与训练数据隔离）
    if eval_data:
        eval_path = os.path.join(args.output_dir, "ifeval_test.json")
        eval_alpaca = [{"instruction": d["instruction"], "input": "", "output": ""} for d in eval_data]
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_alpaca, f, ensure_ascii=False, indent=2)
        print(f"已保存评估集: {eval_path}（{len(eval_alpaca)} 条）")
    
    # 统计
    print("\n" + "=" * 50)
    print("数据集统计")
    print("=" * 50)
    print(f"训练数据总量: {len(all_data)}")
    
    # 按来源统计
    source_stats = {}
    lang_stats = {}
    for item in all_data:
        source = item.get("source", "unknown")
        lang = item.get("language", "unknown")
        source_stats[source] = source_stats.get(source, 0) + 1
        lang_stats[lang] = lang_stats.get(lang, 0) + 1
    
    print("\n数据来源:")
    for source, count in sorted(source_stats.items()):
        print(f"  - {source}: {count}")
    
    print("\n语言分布:")
    for lang, count in sorted(lang_stats.items()):
        print(f"  - {lang}: {count}")
    
    # 统计有响应的数量
    with_output = sum(1 for d in all_data if d.get("output"))
    print(f"\n带响应数据: {with_output} / {len(all_data)}")
    
    if alpaca_data:
        print(f"可用于训练: {len(alpaca_data)}")
    
    if eval_data:
        print(f"评估集: {len(eval_data)}")


if __name__ == "__main__":
    main()
