#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练监控脚本 - 实时监控训练进度和状态
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path


def parse_trainer_log(log_file):
    """解析训练日志"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        return None
    
    # 获取最后一条有效记录
    for line in reversed(lines):
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except:
                continue
    return None


def format_time(seconds_str):
    """格式化时间字符串"""
    if not seconds_str or seconds_str == "0:00:00":
        return seconds_str
    
    # 解析时间字符串
    parts = seconds_str.split(', ')
    if len(parts) == 2:
        # "1 day, 6:03:36" 格式
        days = int(parts[0].split()[0])
        time_parts = parts[1].split(':')
    else:
        # "6:03:36" 格式
        days = 0
        time_parts = parts[0].split(':')
    
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    
    if days > 0:
        return f"{days}天{hours}小时{minutes}分"
    elif hours > 0:
        return f"{hours}小时{minutes}分"
    else:
        return f"{minutes}分钟"


def check_checkpoints(output_dir):
    """检查已保存的检查点"""
    checkpoints = []
    if not os.path.exists(output_dir):
        return checkpoints
    
    for item in os.listdir(output_dir):
        if item.startswith('checkpoint-'):
            checkpoint_path = os.path.join(output_dir, item)
            if os.path.isdir(checkpoint_path):
                step = int(item.split('-')[1])
                checkpoints.append({
                    'step': step,
                    'path': item,
                    'time': datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
                })
    
    return sorted(checkpoints, key=lambda x: x['step'])


def display_progress(log_data, checkpoints, output_dir):
    """显示训练进度"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 70)
    print("🚀 Granite 4.0-1B LoRA 微调监控")
    print("=" * 70)
    print(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not log_data:
        print("⏳ 等待训练开始...\n")
        print(f"输出目录: {output_dir}")
        return
    
    # 训练进度
    current_step = log_data.get('current_steps', 0)
    total_steps = log_data.get('total_steps', 564)
    percentage = log_data.get('percentage', 0)
    epoch = log_data.get('epoch', 0)
    
    print("📊 训练进度")
    print("-" * 70)
    print(f"当前步数: {current_step}/{total_steps}")
    print(f"完成度: {percentage:.1f}%")
    print(f"当前轮次: {epoch:.2f}/3.0")
    
    # 进度条
    bar_length = 50
    filled = int(bar_length * percentage / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"[{bar}] {percentage:.1f}%\n")
    
    # 训练指标
    loss = log_data.get('loss', 0)
    lr = log_data.get('lr', 0)
    
    print("📈 训练指标")
    print("-" * 70)
    print(f"Loss: {loss:.4f}")
    print(f"学习率: {lr:.2e}\n")
    
    # 时间统计
    elapsed = log_data.get('elapsed_time', '0:00:00')
    remaining = log_data.get('remaining_time', '0:00:00')
    
    print("⏱️  时间统计")
    print("-" * 70)
    print(f"已用时间: {format_time(elapsed)}")
    print(f"预计剩余: {format_time(remaining)}\n")
    
    # 检查点信息
    if checkpoints:
        print("💾 检查点保存")
        print("-" * 70)
        print(f"已保存: {len(checkpoints)} 个检查点")
        print(f"最新: checkpoint-{checkpoints[-1]['step']} ({checkpoints[-1]['time'].strftime('%H:%M:%S')})")
        print(f"检查点列表: {', '.join([f'step-{c['step']}' for c in checkpoints[-5:]])}\n")
    else:
        print("💾 检查点保存")
        print("-" * 70)
        print("暂无检查点（每 100 步保存一次）\n")
    
    # 下次检查点预告
    next_checkpoint = ((current_step // 100) + 1) * 100
    if next_checkpoint <= total_steps:
        steps_to_checkpoint = next_checkpoint - current_step
        print(f"📍 下次检查点: step-{next_checkpoint} (还需 {steps_to_checkpoint} 步)")
    
    print("=" * 70)
    print("按 Ctrl+C 停止监控（不会影响训练）")


def monitor(output_dir, interval=30):
    """监控训练过程"""
    log_file = os.path.join(output_dir, 'trainer_log.jsonl')
    
    print("开始监控训练进度...")
    print(f"日志文件: {log_file}")
    print(f"刷新间隔: {interval} 秒\n")
    
    try:
        while True:
            log_data = parse_trainer_log(log_file)
            checkpoints = check_checkpoints(output_dir)
            display_progress(log_data, checkpoints, output_dir)
            
            # 检查是否训练完成
            if log_data and log_data.get('current_steps', 0) >= log_data.get('total_steps', 564):
                print("\n✅ 训练已完成！")
                break
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n监控已停止（训练仍在后台运行）")


def main():
    parser = argparse.ArgumentParser(description="监控 LoRA 微调训练进度")
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs/granite-4.0-1B-lora_v4_optimal',
        help='训练输出目录'
    )
    parser.add_argument(
        '--interval', 
        type=int, 
        default=30,
        help='刷新间隔（秒）'
    )
    args = parser.parse_args()
    
    monitor(args.output_dir, args.interval)


if __name__ == "__main__":
    main()
