#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI数据集生成脚本 - 生成中英翻译和文章总结的混合训练数据
支持OpenAI兼容API（GPT、DeepSeek、Qwen、Gemini等）

设计原则：
A. 指令多样性 - 每种任务使用多种不同的指令表述
B. 数据配比 - 翻译:总结 = 1:1 或 2:1，保持平衡
C. 统一领域 - 聚焦计算机/AI领域，避免跨领域灾难性遗忘

输出格式：LLaMA-Factory alpaca格式
"""

import os
import json
import random
import time
import argparse
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================================
# A. 指令多样性：每种任务定义多种不同的指令模板
# ============================================================================

# 英译中指令模板
EN2ZH_INSTRUCTIONS = [
    "将下面这段计算机领域的学术摘要翻译成中文。",
    "Translate the following CS paper abstract into Chinese.",
    "请把以下技术文档翻译成中文。",
    "Help me translate this technical content to Chinese.",
    "中文翻译：",
    "把这段话翻成中文。",
    "请将以下英文内容译为中文，保持专业术语准确。",
    "Translate to Chinese:",
    "请翻译以下AI领域的文本。",
    "将下文翻译成中文，注意保留技术术语。",
]

# 中译英指令模板
ZH2EN_INSTRUCTIONS = [
    "将以下中文技术文档翻译成英文。",
    "Translate the following Chinese text into English.",
    "请把这段中文翻译成英文。",
    "English translation:",
    "请将以下内容翻译成英文，保持学术风格。",
    "Help me translate this to English.",
    "翻译成英文：",
    "将下文译为英文，注意专业术语的准确性。",
    "请将以下AI相关内容翻译成英文。",
    "Translate to English:",
]

# 总结任务指令模板
SUMMARIZATION_INSTRUCTIONS = [
    "请提取以下论文片段的核心观点，并进行简要总结。",
    "阅读下文，用一句话概括其主要贡献。",
    "Summarize the key points of the following technical content.",
    "请总结以下技术文档的核心内容。",
    "用简洁的语言概括下文的主要观点。",
    "Extract the main ideas from the following text.",
    "请对以下内容进行摘要，突出关键信息。",
    "简要总结以下技术文章的核心论点。",
    "What are the main points of the following content?",
    "请提炼以下段落的核心思想。",
    "概括下文的主要内容和贡献。",
    "Briefly summarize the following passage.",
]

# ============================================================================
# C. 统一领域：聚焦计算机/AI领域的话题
# ============================================================================

CS_AI_TOPICS = [
    # 深度学习
    "Transformer架构的优化", "注意力机制的改进", "大语言模型的训练方法",
    "模型压缩与量化技术", "知识蒸馏方法", "神经网络剪枝策略",
    "模型微调技术", "参数高效微调(PEFT)", "LoRA适配器",
    
    # 自然语言处理
    "文本生成技术", "机器翻译方法", "命名实体识别",
    "情感分析算法", "问答系统设计", "文档摘要生成",
    "预训练语言模型", "Prompt Engineering", "上下文学习",
    
    # 计算机视觉
    "图像分类网络", "目标检测算法", "语义分割方法",
    "图像生成模型", "视觉Transformer", "多模态学习",
    
    # 系统与工程
    "分布式训练框架", "GPU并行计算", "模型部署优化",
    "边缘计算推理", "模型服务化", "MLOps实践",
    "数据流水线设计", "特征工程方法", "模型监控与维护",
    
    # 算法与理论
    "优化器算法改进", "正则化技术", "过拟合与欠拟合",
    "损失函数设计", "激活函数选择", "批归一化技术",
    "学习率调度策略", "梯度裁剪方法", "权重初始化",
    
    # 前沿研究
    "自监督学习", "强化学习与人类反馈(RLHF)", "联邦学习",
    "持续学习方法", "小样本学习", "零样本学习",
    "模型可解释性", "AI安全与对齐", "幻觉问题与缓解",
]


class DatasetGenerator:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        max_workers: int = 5,
        retry_times: int = 3,
    ):
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        self.model = model
        self.max_workers = max_workers
        self.retry_times = retry_times

    def _call_api(self, messages: List[Dict], temperature: float = 0.8) -> Optional[str]:
        """调用API并处理重试"""
        for attempt in range(self.retry_times):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2048,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.retry_times - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    print(f"API调用失败: {e}")
                    return None
        return None

    def generate_translation_pair(self, direction: str, topic: str) -> Optional[Dict]:
        """生成翻译数据对 - 使用多样化指令"""
        if direction == "zh2en":
            source_lang, target_lang = "中文", "英文"
            instruction = random.choice(ZH2EN_INSTRUCTIONS)
        else:  # en2zh
            source_lang, target_lang = "英文", "中文"
            instruction = random.choice(EN2ZH_INSTRUCTIONS)

        # 生成计算机/AI领域的源文本
        gen_prompt = f"""请用{source_lang}写一段关于"{topic}"的专业技术段落。
要求：
1. 长度：150-300字
2. 内容专业、技术性强，适合学术或技术文档
3. 使用该领域的专业术语
4. 逻辑清晰，有技术深度

只输出段落内容，不要有任何额外说明。"""

        source_text = self._call_api([
            {"role": "system", "content": f"你是一位计算机科学和人工智能领域的{source_lang}技术写作专家。"},
            {"role": "user", "content": gen_prompt}
        ])
        
        if not source_text:
            return None

        # 生成高质量翻译
        trans_prompt = f"""请将以下{source_lang}技术文本翻译成{target_lang}：

{source_text}

要求：
1. 翻译准确，保留原文技术含义
2. 专业术语翻译规范（如：Transformer、attention、gradient等保留英文或使用标准译法）
3. 译文自然流畅
4. 保持技术文档的严谨风格

只输出翻译结果，不要有任何额外说明。"""

        translation = self._call_api([
            {"role": "system", "content": f"你是一位计算机科学领域的资深{source_lang}-{target_lang}翻译专家。"},
            {"role": "user", "content": trans_prompt}
        ])

        if not translation:
            return None

        return {
            "instruction": instruction,
            "input": source_text,
            "output": translation,
            "task_type": "translation",
            "direction": direction,
            "topic": topic,
        }

    def generate_summarization_sample(self, topic: str) -> Optional[Dict]:
        """生成文章总结数据 - 使用多样化指令，统一领域"""
        # 随机选择语言（保持双语能力）
        lang = random.choice(["中文", "英文"])
        instruction = random.choice(SUMMARIZATION_INSTRUCTIONS)
        
        # 生成计算机/AI领域的技术文章
        gen_prompt = f"""请用{lang}写一段关于"{topic}"的技术文章内容。
要求：
1. 长度：300-600字
2. 内容专业，有技术深度
3. 结构清晰，包含问题背景、方法/方案、效果/结论等要素
4. 使用该领域的专业术语
5. 适合进行摘要提取

只输出文章内容，不要有标题或额外说明。"""

        article = self._call_api([
            {"role": "system", "content": f"你是一位计算机科学和人工智能领域的{lang}技术作者。"},
            {"role": "user", "content": gen_prompt}
        ])

        if not article:
            return None

        # 根据指令风格生成对应的总结
        # 判断指令是否要求"一句话概括"
        is_one_sentence = "一句话" in instruction or "main points" in instruction.lower()
        
        if is_one_sentence:
            summary_prompt = f"""请用一句话概括以下技术内容的核心观点或主要贡献：

{article}

要求：
1. 只用一句话，不超过50字
2. 抓住最核心的技术贡献或观点
3. 语言精炼准确

只输出总结，不要有任何额外说明。"""
        else:
            summary_prompt = f"""请对以下技术内容进行简要总结：

{article}

要求：
1. 总结长度：50-120字
2. 提取核心观点和关键技术信息
3. 语言精炼，逻辑清晰
4. 保持技术准确性

只输出总结内容，不要有任何额外说明。"""

        summary = self._call_api([
            {"role": "system", "content": "你是一位专业的技术文档分析和总结专家。"},
            {"role": "user", "content": summary_prompt}
        ])

        if not summary:
            return None

        return {
            "instruction": instruction,
            "input": article,
            "output": summary,
            "task_type": "summarization",
            "topic": topic,
            "language": lang,
        }

    def generate_dataset(
        self,
        total_samples: int = 2000,
        translation_ratio: float = 0.5,
        output_file: str = "data/train.json",
    ) -> List[Dict]:
        """
        生成混合数据集
        
        B. 数据配比：translation_ratio 控制翻译占比
           - 0.5 = 翻译:总结 = 1:1 (默认，推荐)
           - 0.67 = 翻译:总结 = 2:1
        """
        num_translation = int(total_samples * translation_ratio)
        num_summarization = total_samples - num_translation
        
        # 分配翻译任务（中译英和英译中各一半）
        num_zh2en = num_translation // 2
        num_en2zh = num_translation - num_zh2en
        
        print(f"=" * 60)
        print(f"数据集生成计划 (聚焦计算机/AI领域)")
        print(f"=" * 60)
        print(f"  翻译数据: {num_translation}条")
        print(f"    - 中译英: {num_zh2en}条")
        print(f"    - 英译中: {num_en2zh}条")
        print(f"  总结数据: {num_summarization}条")
        print(f"  配比: 翻译:总结 = {num_translation}:{num_summarization}")
        print(f"  总计: {total_samples}条")
        print(f"=" * 60)
        print()

        results = []
        failed = 0
        
        # 生成任务列表
        tasks = []
        for _ in range(num_zh2en):
            topic = random.choice(CS_AI_TOPICS)
            tasks.append(("translation", "zh2en", topic))
        for _ in range(num_en2zh):
            topic = random.choice(CS_AI_TOPICS)
            tasks.append(("translation", "en2zh", topic))
        for _ in range(num_summarization):
            topic = random.choice(CS_AI_TOPICS)
            tasks.append(("summarization", topic, None))
        
        random.shuffle(tasks)

        # 并行执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for task_type, param1, param2 in tasks:
                if task_type == "translation":
                    future = executor.submit(self.generate_translation_pair, param1, param2)
                else:
                    future = executor.submit(self.generate_summarization_sample, param1)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc="生成数据"):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed += 1

        print(f"\n生成完成: 成功 {len(results)}条, 失败 {failed}条")

        # 统计指令多样性
        instruction_stats = {}
        for item in results:
            inst = item["instruction"][:20] + "..."  # 截断显示
            instruction_stats[inst] = instruction_stats.get(inst, 0) + 1
        
        print(f"\n指令分布统计 (前10种):")
        for inst, count in sorted(instruction_stats.items(), key=lambda x: -x[1])[:10]:
            print(f"  {inst}: {count}条")

        # 转换为LLaMA-Factory alpaca格式
        alpaca_data = []
        for item in results:
            alpaca_item = {
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"],
            }
            alpaca_data.append(alpaca_item)

        # 随机打乱
        random.shuffle(alpaca_data)

        # 划分训练集和验证集 (9:1)
        split_idx = int(len(alpaca_data) * 0.9)
        train_data = alpaca_data[:split_idx]
        val_data = alpaca_data[split_idx:]

        # 保存数据
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        train_file = output_file
        val_file = output_file.replace("train.json", "val.json")
        
        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        print(f"\n数据已保存:")
        print(f"  - 训练集: {train_file} ({len(train_data)}条)")
        print(f"  - 验证集: {val_file} ({len(val_data)}条)")

        # 保存完整数据（含元信息）用于分析
        full_file = output_file.replace("train.json", "full_with_meta.json")
        with open(full_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  - 完整数据(含元信息): {full_file}")

        return alpaca_data


def main():
    parser = argparse.ArgumentParser(description="生成微调数据集（计算机/AI领域）")
    parser.add_argument("--total", type=int, default=2000, help="总样本数")
    parser.add_argument(
        "--translation-ratio", type=float, default=0.5,
        help="翻译数据占比 (0.5=1:1配比, 0.67=2:1配比)"
    )
    parser.add_argument("--output", type=str, default="data/train.json", help="输出文件路径")
    parser.add_argument("--workers", type=int, default=5, help="并行工作线程数")
    args = parser.parse_args()

    # 从环境变量读取API配置
    api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("API_KEY", "")
    model = os.getenv("API_MODEL", "gpt-4o")

    if not api_key:
        print("错误: 请设置API_KEY环境变量或在.env文件中配置")
        return

    print(f"API配置:")
    print(f"  - Base URL: {api_base}")
    print(f"  - Model: {model}")
    print()

    generator = DatasetGenerator(
        api_base=api_base,
        api_key=api_key,
        model=model,
        max_workers=args.workers,
    )

    generator.generate_dataset(
        total_samples=args.total,
        translation_ratio=args.translation_ratio,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
