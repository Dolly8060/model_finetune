#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估脚本 - 对比微调前后以及不同规模模型在翻译和总结任务上的表现
评估指标：BLEU、ROUGE、BERTScore（业界标准）

支持功能：
1. 基础模型 vs 微调模型对比
2. 多模型横向对比（如1B vs 7B vs 8B）
"""

import os
import json
import argparse
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from dotenv import load_dotenv

# 评估指标
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score
import jieba

load_dotenv()


# 预定义模型的prompt模板
MODEL_TEMPLATES = {
    # Granite系列
    "granite": {
        "format": "<|start_of_role|>user<|end_of_role|>{instruction}\n\n{input}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
    },
    # Qwen系列
    "qwen": {
        "format": "<|im_start|>user\n{instruction}\n\n{input}<|im_end|>\n<|im_start|>assistant\n",
    },
    # LLaMA-3系列
    "llama": {
        "format": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    # 通用ChatML格式
    "chatml": {
        "format": "<|im_start|>user\n{instruction}\n\n{input}<|im_end|>\n<|im_start|>assistant\n",
    },
    # 默认格式（简单拼接）
    "default": {
        "format": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    },
}


@dataclass
class EvalResult:
    """评估结果"""
    bleu: float
    rouge1: float
    rouge2: float
    rougeL: float
    bert_precision: float
    bert_recall: float
    bert_f1: float


def detect_model_type(model_name: str) -> str:
    """根据模型名称检测模型类型，返回对应的模板key"""
    model_name_lower = model_name.lower()
    
    if "granite" in model_name_lower:
        return "granite"
    elif "qwen" in model_name_lower:
        return "qwen"
    elif "llama" in model_name_lower:
        return "llama"
    elif "gemma" in model_name_lower:
        return "chatml"  # Gemma使用类似ChatML的格式
    else:
        return "default"


def format_prompt(instruction: str, input_text: str, model_type: str) -> str:
    """根据模型类型格式化prompt"""
    template = MODEL_TEMPLATES.get(model_type, MODEL_TEMPLATES["default"])
    return template["format"].format(instruction=instruction, input=input_text)


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: str = None,
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """
    加载模型和tokenizer
    
    Returns:
        model, tokenizer, model_type
    """
    print(f"加载模型: {model_path}")
    
    # 检测模型类型
    model_type = detect_model_type(model_path)
    print(f"  检测到模型类型: {model_type}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # 如果有adapter，加载LoRA
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


class ModelEvaluator:
    def __init__(
        self,
        base_model_path: str,
        finetuned_model_path: str = None,
        adapter_path: str = None,
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        """
        初始化评估器
        
        Args:
            base_model_path: 基础模型路径
            finetuned_model_path: 全量微调后的模型路径（与adapter_path二选一）
            adapter_path: LoRA adapter路径
            device: 设备
            max_new_tokens: 生成最大token数
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        print("加载Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        print("加载基础模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.base_model.eval()
        
        # 保存adapter路径，延迟加载（避免影响base_model评估）
        self.finetuned_model = None
        self.adapter_path = adapter_path
        self.finetuned_model_path = finetuned_model_path
        self.base_model_path = base_model_path
        
        # 初始化评估器
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.bleu = BLEU(effective_order=True, lowercase=False, tokenize='zh')

    def _generate(self, model, prompt: str) -> str:
        """使用模型生成回复"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1536
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # 使用贪婪解码确保可复现
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 只取新生成的部分
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def _format_prompt(self, instruction: str, input_text: str) -> str:
        """格式化prompt（Granite模板）"""
        # Granite使用的对话模板
        prompt = f"""<|start_of_role|>user<|end_of_role|>{instruction}

{input_text}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>"""
        return prompt

    def _tokenize_chinese(self, text: str) -> List[str]:
        """中文分词（用于BLEU计算）"""
        return list(jieba.cut(text))

    def _detect_lang(self, text: str) -> str:
        """检测文本语言（中文/英文/混合）"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return "en"
        ratio = chinese_chars / total_chars
        if ratio > 0.3:
            return "zh"
        return "en"

    def _classify_sample(self, sample: Dict) -> Optional[str]:
        """根据instruction或task_type粗略判断是翻译还是总结"""
        task_type = sample.get("task_type", "").lower()
        if task_type in ("translation", "summarization"):
            return task_type
        instr = sample.get("instruction", "")
        instr_lower = instr.lower()
        translation_keywords = [
            "翻译", "translate", "translation", "译为", "译成",
            "english translation", "中文翻译", "中译英", "英译中",
        ]
        summary_keywords = [
            "总结", "摘要", "概括", "summary", "summarize",
            "提炼", "提取主要观点", "总结以下", "概括下文",
        ]
        if any(kw in instr_lower for kw in translation_keywords):
            return "translation"
        if any(kw in instr_lower for kw in summary_keywords):
            return "summarization"
        return None

    def _compute_metrics(
        self, predictions: List[str], references: List[str], lang: str = "mixed"
    ) -> EvalResult:
        """计算评估指标"""
        # 处理空预测
        predictions = [p.strip() if p else " " for p in predictions]
        references = [r.strip() if r else " " for r in references]
        
        # 清理预测中的常见前缀（模型可能生成的提示性文本）
        cleaned_predictions = []
        for pred in predictions:
            # 移除常见前缀
            for prefix in [
                "Here is the English translation:",
                "Here is the Chinese translation:",
                "Translation:",
                "翻译：",
                "翻译结果：",
                "译文：",
            ]:
                if pred.startswith(prefix):
                    pred = pred[len(prefix):].strip()
                    break
            cleaned_predictions.append(pred)
        
        predictions = cleaned_predictions
        
        # 自动检测语言（基于参考答案）
        sample_text = ''.join(references[:10])
        detected_lang = self._detect_lang(sample_text)
        
        # BLEU（使用SacreBLEU的自动分词）
        try:
            if detected_lang == "zh":
                # 中文：使用字符级分词
                bleu_score = self.bleu.corpus_score(predictions, [[r] for r in references]).score
            else:
                # 英文：SacreBLEU自动处理
                bleu_score = self.bleu.corpus_score(predictions, [[r] for r in references]).score
        except Exception as e:
            print(f"BLEU计算失败: {e}, 返回0")
            bleu_score = 0.0
        
        # ROUGE
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        # BERTScore（自动检测语言）
        P, R, F1 = bert_score(
            predictions, references, 
            lang=detected_lang,
            verbose=False
        )
        
        return EvalResult(
            bleu=bleu_score,
            rouge1=np.mean(rouge1_scores) * 100,
            rouge2=np.mean(rouge2_scores) * 100,
            rougeL=np.mean(rougeL_scores) * 100,
            bert_precision=P.mean().item() * 100,
            bert_recall=R.mean().item() * 100,
            bert_f1=F1.mean().item() * 100,
        )

    def evaluate_model(
        self, model, samples: List[Dict], desc: str = "评估"
    ) -> Tuple[EvalResult, List[Dict]]:
        """评估单个模型"""
        predictions = []
        references = []
        detailed_results = []
        
        for sample in tqdm(samples, desc=desc):
            prompt = self._format_prompt(sample["instruction"], sample["input"])
            prediction = self._generate(model, prompt)
            predictions.append(prediction)
            references.append(sample["output"])
            
            detailed_results.append({
                "instruction": sample["instruction"],
                "input": sample["input"][:200] + "..." if len(sample["input"]) > 200 else sample["input"],
                "reference": sample["output"],
                "prediction": prediction,
            })
        
        metrics = self._compute_metrics(predictions, references)
        return metrics, detailed_results

    def compare_models(
        self, eval_file: str, max_samples: int = 0, output_dir: str = "evaluation"
    ) -> Dict:
        """对比微调前后的模型表现（按任务类型拆分：翻译/总结）"""
        # 加载评估数据
        print(f"加载评估数据: {eval_file}")
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        # 随机采样（保证可复现）；max_samples=0 表示全量
        if max_samples and len(eval_data) > max_samples:
            random.seed(42)
            eval_data = random.sample(eval_data, max_samples)
        print(f"评估样本数: {len(eval_data)}")
        
        # 按任务类型拆分
        translation_data: List[Dict] = []
        summarization_data: List[Dict] = []
        for sample in eval_data:
            task = self._classify_sample(sample)
            if task == "translation":
                translation_data.append(sample)
            elif task == "summarization":
                summarization_data.append(sample)
        
        print(f"翻译样本数: {len(translation_data)}")
        print(f"总结样本数: {len(summarization_data)}")
        
        results: Dict[str, Dict] = {}
        
        # 评估基础模型
        print("\n" + "=" * 50)
        print("评估基础模型")
        print("=" * 50)
        
        if translation_data:
            base_metrics_tr, base_details_tr = self.evaluate_model(
                self.base_model, translation_data, "基础模型-翻译子集"
            )
            results.setdefault("translation", {})["base_model"] = {
                "metrics": base_metrics_tr.__dict__,
                "details": base_details_tr,
            }
        
        if summarization_data:
            base_metrics_sum, base_details_sum = self.evaluate_model(
                self.base_model, summarization_data, "基础模型-总结子集"
            )
            results.setdefault("summarization", {})["base_model"] = {
                "metrics": base_metrics_sum.__dict__,
                "details": base_details_sum,
            }
        
        # 释放基础模型显存（如果需要加载微调模型）
        if self.finetuned_model_path or self.adapter_path:
            print("\n释放基础模型显存...")
            del self.base_model
            gc.collect()
            torch.cuda.empty_cache()
        
        # 加载微调模型
        if self.finetuned_model_path:
            print("\n加载全量微调模型...")
            self.finetuned_model = AutoModelForCausalLM.from_pretrained(
                self.finetuned_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.finetuned_model.eval()
        elif self.adapter_path:
            print("\n加载LoRA适配器...")
            finetuned_base = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.finetuned_model = PeftModel.from_pretrained(
                finetuned_base, self.adapter_path
            )
            self.finetuned_model.eval()
        
        # 评估微调模型
        if self.finetuned_model:
            print("\n" + "=" * 50)
            print("评估微调模型")
            print("=" * 50)
            if translation_data:
                ft_metrics_tr, ft_details_tr = self.evaluate_model(
                    self.finetuned_model, translation_data, "微调模型-翻译子集"
                )
                results.setdefault("translation", {})["finetuned_model"] = {
                    "metrics": ft_metrics_tr.__dict__,
                    "details": ft_details_tr,
                }
            if summarization_data:
                ft_metrics_sum, ft_details_sum = self.evaluate_model(
                    self.finetuned_model, summarization_data, "微调模型-总结子集"
                )
                results.setdefault("summarization", {})["finetuned_model"] = {
                    "metrics": ft_metrics_sum.__dict__,
                    "details": ft_details_sum,
                }
        
        # 生成对比报告
        self._generate_report(results, output_dir)
        
        return results

    def _generate_report(self, results: Dict, output_dir: str):
        """生成评估报告（按任务类型拆分）"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存指标摘要（按子集/任务类型）
        summary = {}
        for subset_name, subset_res in results.items():
            summary[subset_name] = {}
            if "base_model" in subset_res:
                summary[subset_name]["base_model"] = subset_res["base_model"]["metrics"]
            if "finetuned_model" in subset_res:
                summary[subset_name]["finetuned_model"] = subset_res["finetuned_model"]["metrics"]
        with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 保存详细结果（用于人工复核）
        details_out = {}
        for subset_name, subset_res in results.items():
            details_out[subset_name] = {}
            if "base_model" in subset_res:
                details_out[subset_name]["base_model"] = subset_res["base_model"]["details"][:20]
            if "finetuned_model" in subset_res:
                details_out[subset_name]["finetuned_model"] = subset_res["finetuned_model"]["details"][:20]
        with open(os.path.join(output_dir, "eval_details.json"), "w", encoding="utf-8") as f:
            json.dump(details_out, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存: {os.path.join(output_dir, 'eval_details.json')} (各子集前20条样本)")
        
        # 生成Markdown报告
        report = ["# 模型评估报告\n"]
        report.append("## 评估指标说明\n")
        report.append("- **BLEU**: 机器翻译标准评估指标，主要用于翻译子集")
        report.append("- **ROUGE-1/2/L**: 文本摘要评估指标，主要用于总结子集")
        report.append("- **BERTScore**: 基于BERT的语义相似度评估\n")
        
        metrics_names = [
            ("BLEU", "bleu"),
            ("ROUGE-1", "rouge1"),
            ("ROUGE-2", "rouge2"),
            ("ROUGE-L", "rougeL"),
            ("BERTScore-P", "bert_precision"),
            ("BERTScore-R", "bert_recall"),
            ("BERTScore-F1", "bert_f1"),
        ]
        
        subset_labels = {
            "translation": "翻译子集 (Translation)",
            "summarization": "总结子集 (Summarization)",
        }
        
        for subset_name, label in subset_labels.items():
            if subset_name not in results:
                continue
            subset_res = results[subset_name]
            base_m = subset_res["base_model"]["metrics"]
            ft_m = subset_res.get("finetuned_model", {}).get("metrics", {})
            
            report.append(f"\n## {label}\n")
            report.append("| 指标 | 基础模型 | 微调模型 | 提升 |")
            report.append("|------|----------|----------|------|")
            
            for display_name, key in metrics_names:
                base_val = base_m.get(key, 0)
                ft_val = ft_m.get(key, 0) if ft_m else "-"
                if ft_m:
                    diff = ft_val - base_val
                    diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                else:
                    diff_str = "-"
                ft_str = f"{ft_val:.2f}" if isinstance(ft_val, (int, float)) else ft_val
                report.append(f"| {display_name} | {base_val:.2f} | {ft_str} | {diff_str} |")
            
            report.append("\n### 子集结论\n")
            if ft_m:
                improvements = []
                for display_name, key in metrics_names:
                    diff = ft_m.get(key, 0) - base_m.get(key, 0)
                    if diff > 0:
                        improvements.append(f"{display_name} (+{diff:.2f})")
                if improvements:
                    report.append(f"微调后模型在该子集的以下指标上有提升: {', '.join(improvements)}")
                else:
                    report.append("微调后模型在该子集的主要指标上未见明显提升，建议调整训练参数或增加数据量。")
            else:
                report.append("仅评估了基础模型，请提供微调后的模型进行对比。")
        
        report_path = os.path.join(output_dir, "eval_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"\n评估报告已保存: {report_path}")
        
        # 打印摘要（终端）
        print("\n" + "=" * 60)
        print("评估结果摘要（按子集）")
        print("=" * 60)
        for subset_name, label in subset_labels.items():
            if subset_name not in results:
                continue
            subset_res = results[subset_name]
            base_m = subset_res["base_model"]["metrics"]
            ft_m = subset_res.get("finetuned_model", {}).get("metrics", {})
            print(f"\n[{label}]")
            print(f"{'指标':<15} {'基础模型':>12} {'微调模型':>12} {'提升':>10}")
            print("-" * 60)
            for display_name, key in metrics_names:
                base_val = base_m.get(key, 0)
                ft_val = ft_m.get(key, 0) if ft_m else 0
                diff = ft_val - base_val if ft_m else 0
                diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                ft_str = f"{ft_val:.2f}" if ft_m else "-"
                print(f"{display_name:<15} {base_val:>12.2f} {ft_str:>12} {diff_str:>10}")
        print("=" * 60)


class MultiModelEvaluator:
    """多模型对比评估器"""
    
    def __init__(self, max_new_tokens: int = 512):
        self.max_new_tokens = max_new_tokens
        self.rouge_scorer_obj = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.bleu = BLEU(effective_order=True, lowercase=False, tokenize='zh')
    
    def _generate(self, model, tokenizer, prompt: str) -> str:
        """使用模型生成回复"""
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1536
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()
    
    def _tokenize_chinese(self, text: str) -> List[str]:
        return list(jieba.cut(text))
    
    def _detect_lang(self, text: str) -> str:
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return "en"
        return "zh" if chinese_chars / total_chars > 0.3 else "en"
    
    def _compute_metrics(self, predictions: List[str], references: List[str]) -> EvalResult:
        """计算评估指标"""
        # 处理空预测
        predictions = [p.strip() if p else " " for p in predictions]
        references = [r.strip() if r else " " for r in references]
        
        # 清理预测中的常见前缀
        cleaned_predictions = []
        for pred in predictions:
            for prefix in [
                "Here is the English translation:",
                "Here is the Chinese translation:",
                "Translation:",
                "翻译：",
                "翻译结果：",
                "译文：",
            ]:
                if pred.startswith(prefix):
                    pred = pred[len(prefix):].strip()
                    break
            cleaned_predictions.append(pred)
        predictions = cleaned_predictions
        
        # 检测语言
        sample_text = ''.join(references[:10])
        detected_lang = self._detect_lang(sample_text)
        
        # BLEU（修复：使用SacreBLEU的中文分词）
        try:
            if detected_lang == "zh":
                # 中文：使用字符级分词（SacreBLEU的zh tokenizer）
                bleu_score = self.bleu.corpus_score(predictions, [[r] for r in references]).score
            else:
                # 英文：SacreBLEU自动处理
                bleu_score = self.bleu.corpus_score(predictions, [[r] for r in references]).score
        except Exception as e:
            print(f"BLEU计算失败: {e}, 返回0")
            bleu_score = 0.0
        
        # ROUGE
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer_obj.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        # BERTScore
        P, R, F1 = bert_score(predictions, references, lang=detected_lang, verbose=False)
        
        return EvalResult(
            bleu=bleu_score,
            rouge1=np.mean(rouge1_scores) * 100,
            rouge2=np.mean(rouge2_scores) * 100,
            rougeL=np.mean(rougeL_scores) * 100,
            bert_precision=P.mean().item() * 100,
            bert_recall=R.mean().item() * 100,
            bert_f1=F1.mean().item() * 100,
        )
    
    def _classify_sample(self, sample: Dict) -> Optional[str]:
        """根据instruction或task_type粗略判断是翻译还是总结"""
        task_type = sample.get("task_type", "").lower()
        if task_type in ("translation", "summarization"):
            return task_type
        instr = sample.get("instruction", "")
        instr_lower = instr.lower()
        translation_keywords = [
            "翻译", "translate", "translation", "译为", "译成",
            "english translation", "中文翻译", "中译英", "英译中",
        ]
        summary_keywords = [
            "总结", "摘要", "概括", "summary", "summarize",
            "提炼", "提取主要观点", "总结以下", "概括下文",
        ]
        if any(kw in instr_lower for kw in translation_keywords):
            return "translation"
        if any(kw in instr_lower for kw in summary_keywords):
            return "summarization"
        return None
    
    def evaluate_single_model(
        self,
        model,
        tokenizer,
        model_type: str,
        samples: List[Dict],
        model_name: str,
        desc_suffix: str = "",
    ) -> Tuple[EvalResult, List[Dict]]:
        """评估单个模型"""
        predictions = []
        references = []
        detailed_results = []
        
        desc = f"评估 {model_name}{desc_suffix}"
        for sample in tqdm(samples, desc=desc):
            prompt = format_prompt(sample["instruction"], sample["input"], model_type)
            prediction = self._generate(model, tokenizer, prompt)
            predictions.append(prediction)
            references.append(sample["output"])
            
            detailed_results.append({
                "instruction": sample["instruction"],
                "input": sample["input"][:200] + "..." if len(sample["input"]) > 200 else sample["input"],
                "reference": sample["output"],
                "prediction": prediction,
            })
        
        metrics = self._compute_metrics(predictions, references)
        return metrics, detailed_results
    
    def compare_multiple_models(
        self,
        model_configs: List[Dict],
        eval_file: str,
        max_samples: int = 0,
        output_dir: str = "evaluation",
    ) -> Dict:
        """
        对比多个模型（按任务类型拆分：翻译/总结）
        
        Args:
            model_configs: 模型配置列表，每个元素为 {"name": "显示名称", "path": "模型路径", "adapter": "可选的adapter路径"}
            eval_file: 评估数据文件
            max_samples: 最大评估样本数
            output_dir: 输出目录
        """
        # 加载评估数据
        print(f"\n加载评估数据: {eval_file}")
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        if max_samples and len(eval_data) > max_samples:
            random.seed(42)
            eval_data = random.sample(eval_data, max_samples)
        print(f"评估样本数: {len(eval_data)}")
        
        # 按任务类型拆分数据
        translation_data: List[Dict] = []
        summarization_data: List[Dict] = []
        for sample in eval_data:
            task = self._classify_sample(sample)
            if task == "translation":
                translation_data.append(sample)
            elif task == "summarization":
                summarization_data.append(sample)
        
        print(f"翻译样本数: {len(translation_data)}")
        print(f"总结样本数: {len(summarization_data)}")
        
        results: Dict[str, Dict] = {}
        
        # 逐个评估模型
        for i, config in enumerate(model_configs):
            model_name = config["name"]
            model_path = config["path"]
            adapter_path = config.get("adapter")
            
            print(f"\n{'=' * 60}")
            print(f"[{i+1}/{len(model_configs)}] 评估模型: {model_name}")
            print(f"{'=' * 60}")
            
            try:
                # 加载模型
                model, tokenizer, model_type = load_model_and_tokenizer(
                    model_path, adapter_path
                )
                
                # 评估翻译子集
                if translation_data:
                    tr_metrics, tr_details = self.evaluate_single_model(
                        model, tokenizer, model_type, translation_data, model_name, "-翻译子集"
                    )
                    results.setdefault("translation", {})[model_name] = {
                        "path": model_path,
                        "metrics": tr_metrics.__dict__,
                        "details": tr_details,
                    }
                
                # 评估总结子集
                if summarization_data:
                    sum_metrics, sum_details = self.evaluate_single_model(
                        model, tokenizer, model_type, summarization_data, model_name, "-总结子集"
                    )
                    results.setdefault("summarization", {})[model_name] = {
                        "path": model_path,
                        "metrics": sum_metrics.__dict__,
                        "details": sum_details,
                    }
                
                # 释放显存
                print(f"释放 {model_name} 显存...")
                release_model(model, tokenizer)
                
            except Exception as e:
                print(f"评估 {model_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                if translation_data:
                    results.setdefault("translation", {})[model_name] = {
                        "path": model_path,
                        "error": str(e),
                    }
                if summarization_data:
                    results.setdefault("summarization", {})[model_name] = {
                        "path": model_path,
                        "error": str(e),
                    }
        
        # 生成对比报告
        self._generate_comparison_report(results, output_dir)
        
        return results
    
    def _generate_comparison_report(self, results: Dict, output_dir: str):
        """生成多模型对比报告（按任务类型拆分）"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存指标摘要（按子集）
        summary = {}
        for subset_name, subset_res in results.items():
            summary[subset_name] = {}
            for model_name, model_data in subset_res.items():
                if "metrics" in model_data:
                    summary[subset_name][model_name] = model_data["metrics"]
        
        summary_path = os.path.join(output_dir, "comparison_results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n指标摘要已保存: {summary_path}")
        
        # 保存详细结果
        details_out = {}
        for subset_name, subset_res in results.items():
            details_out[subset_name] = {}
            for model_name, model_data in subset_res.items():
                if "details" in model_data:
                    details_out[subset_name][model_name] = model_data["details"][:10]
        
        details_path = os.path.join(output_dir, "comparison_details.json")
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(details_out, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存: {details_path} (每模型每子集前10条)")
        
        # 生成Markdown报告
        report = ["# 多模型对比评估报告\n"]
        report.append("## 评估指标说明\n")
        report.append("- **BLEU**: 机器翻译标准评估指标，主要用于翻译子集")
        report.append("- **ROUGE-1/2/L**: 文本摘要评估指标，主要用于总结子集")
        report.append("- **BERTScore**: 基于BERT的语义相似度评估\n")
        
        metrics_names = [
            ("BLEU", "bleu"),
            ("ROUGE-1", "rouge1"),
            ("ROUGE-2", "rouge2"),
            ("ROUGE-L", "rougeL"),
            ("BERTScore-P", "bert_precision"),
            ("BERTScore-R", "bert_recall"),
            ("BERTScore-F1", "bert_f1"),
        ]
        
        subset_labels = {
            "translation": "翻译子集 (Translation)",
            "summarization": "总结子集 (Summarization)",
        }
        
        # 获取所有模型名称
        all_model_names = set()
        for subset_res in results.values():
            all_model_names.update(subset_res.keys())
        all_model_names = [n for n in all_model_names if not results.get("translation", {}).get(n, {}).get("error")]
        
        # 模型信息表
        if all_model_names:
            report.append("## 模型信息\n")
            report.append("| 模型名称 | 模型路径 |")
            report.append("|----------|----------|")
            for name in all_model_names:
                path = ""
                for subset_res in results.values():
                    if name in subset_res and "path" in subset_res[name]:
                        path = subset_res[name]["path"]
                        break
                report.append(f"| {name} | {path} |")
        
        # 按子集输出对比结果
        for subset_name, label in subset_labels.items():
            if subset_name not in results:
                continue
            
            subset_res = results[subset_name]
            valid_models = [n for n in all_model_names if n in subset_res and "metrics" in subset_res[n]]
            
            if not valid_models:
                continue
            
            report.append(f"\n## {label}\n")
            
            # 构建表头
            header = "| 指标 |"
            separator = "|------|"
            for name in valid_models:
                header += f" {name} |"
                separator += "-------:|"
            report.append(header)
            report.append(separator)
            
            # 输出各指标
            for display_name, key in metrics_names:
                row = f"| {display_name} |"
                for name in valid_models:
                    val = subset_res[name]["metrics"].get(key, 0)
                    row += f" {val:.2f} |"
                report.append(row)
            
            # 找出各指标的最佳模型
            report.append(f"\n### {label} - 最佳模型\n")
            report.append("| 指标 | 最佳模型 | 分数 |")
            report.append("|------|----------|------|")
            
            for display_name, key in metrics_names:
                best_model = max(
                    valid_models,
                    key=lambda n: subset_res[n]["metrics"].get(key, 0)
                )
                best_score = subset_res[best_model]["metrics"].get(key, 0)
                report.append(f"| {display_name} | {best_model} | {best_score:.2f} |")
        
        # 综合结论
        report.append("\n## 综合结论\n")
        
        for subset_name, label in subset_labels.items():
            if subset_name not in results:
                continue
            
            subset_res = results[subset_name]
            valid_models = [n for n in all_model_names if n in subset_res and "metrics" in subset_res[n]]
            
            if not valid_models:
                continue
            
            # 计算子集综合得分
            avg_scores = {}
            for name in valid_models:
                metrics = subset_res[name]["metrics"]
                avg = np.mean([metrics.get(k, 0) for _, k in metrics_names])
                avg_scores[name] = avg
            
            sorted_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            report.append(f"\n**{label}综合排名**：\n")
            for i, (name, score) in enumerate(sorted_models, 1):
                report.append(f"{i}. **{name}**: 平均得分 {score:.2f}")
        
        report_path = os.path.join(output_dir, "comparison_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"对比报告已保存: {report_path}")
        
        # 打印终端摘要
        print("\n" + "=" * 80)
        print("多模型评估结果摘要（按子集）")
        print("=" * 80)
        
        for subset_name, label in subset_labels.items():
            if subset_name not in results:
                continue
            
            subset_res = results[subset_name]
            valid_models = [n for n in all_model_names if n in subset_res and "metrics" in subset_res[n]]
            
            if not valid_models:
                continue
            
            print(f"\n[{label}]")
            
            # 打印表头
            header_line = f"{'指标':<15}"
            for name in valid_models:
                # 截断过长的名称
                display_name = name[:12] if len(name) > 12 else name
                header_line += f"{display_name:>15}"
            print(header_line)
            print("-" * 80)
            
            for display_name, key in metrics_names:
                row = f"{display_name:<15}"
                for name in valid_models:
                    val = subset_res[name]["metrics"].get(key, 0)
                    row += f"{val:>15.2f}"
                print(row)
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="评估模型表现（支持微调对比和多模型对比）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  1. 微调前后对比:
     python evaluate.py --base-model path/to/base --adapter path/to/adapter

  2. 多模型对比:
     python evaluate.py --compare-models \\
       "微调1B:path/to/finetuned" \\
       "Qwen2.5-7B:Qwen/Qwen2.5-7B-Instruct" \\
       "LLaMA-3.1-8B:meta-llama/Llama-3.1-8B-Instruct"

  3. 带adapter的多模型对比:
     python evaluate.py --compare-models \\
       "基础1B:path/to/base" \\
       "微调1B:path/to/base:path/to/adapter"
        """
    )
    
    # 原有参数（微调对比模式）
    parser.add_argument(
        "--base-model", type=str, default=None, help="基础模型路径"
    )
    parser.add_argument(
        "--finetuned-model", type=str, default=None, help="全量微调后的模型路径"
    )
    parser.add_argument(
        "--adapter", type=str, default=None, help="LoRA adapter路径"
    )
    
    # 新增参数（多模型对比模式）
    parser.add_argument(
        "--compare-models", type=str, nargs="+", default=None,
        help="多模型对比，格式: '名称:模型路径' 或 '名称:模型路径:adapter路径'"
    )
    
    # 通用参数
    parser.add_argument(
        "--eval-file", type=str, default="data/val.json", help="评估数据文件"
    )
    parser.add_argument(
        "--max-samples", type=int, default=0, help="最大评估样本数（0表示使用全部样本）"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation", help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 多模型对比模式
    if args.compare_models:
        print("\n" + "=" * 60)
        print("多模型对比评估模式")
        print("=" * 60)
        
        # 解析模型配置（支持Windows路径，如 D:/path）
        def parse_model_spec(spec: str):
            """
            解析模型配置字符串，支持Windows路径
            格式: 名称:模型路径 或 名称:模型路径:adapter路径
            Windows路径示例: 微调模型:D:/models/base:D:/adapters/lora
            """
            # 使用正则匹配，处理Windows驱动器字母
            import re
            # 匹配模式: name : [drive:]/path [: [drive:]/adapter]
            # Windows驱动器是单字母后跟冒号，如 D:
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
        
        model_configs = []
        for model_spec in args.compare_models:
            result = parse_model_spec(model_spec)
            if result is None:
                print(f"错误: 无效的模型配置格式: {model_spec}")
                print("正确格式: '名称:模型路径' 或 '名称:模型路径:adapter路径'")
                print("示例: '微调模型:D:/models/base:outputs/lora'")
                return
            
            name, path, adapter = result
            model_configs.append({
                "name": name,
                "path": path,
                "adapter": adapter,
            })
        
        print(f"\n将评估 {len(model_configs)} 个模型:")
        for config in model_configs:
            adapter_info = f" (adapter: {config['adapter']})" if config['adapter'] else ""
            print(f"  - {config['name']}: {config['path']}{adapter_info}")
        
        evaluator = MultiModelEvaluator()
        evaluator.compare_multiple_models(
            model_configs=model_configs,
            eval_file=args.eval_file,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )
    
    # 原有的微调对比模式
    else:
        base_model = args.base_model or os.getenv("MODEL_PATH")
        if not base_model:
            print("错误: 请指定--base-model参数或设置MODEL_PATH环境变量")
            print("或使用--compare-models进行多模型对比")
            return
        
        evaluator = ModelEvaluator(
            base_model_path=base_model,
            finetuned_model_path=args.finetuned_model,
            adapter_path=args.adapter,
        )
        
        evaluator.compare_models(
            eval_file=args.eval_file,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
