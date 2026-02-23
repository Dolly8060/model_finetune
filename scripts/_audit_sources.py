#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick audit of source data files for Qwen3 dataset building."""
import json, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

files = [
    'data/train.json',
    'data/val.json', 
    'data/ifeval_full_with_meta.json',
    'data/argilla_ifeval.json',
    'data/public_val_v2.json',
    'data/train_v3.json',
    'data/val_v3.json',
    'data/train_mixed_3k.json',
    'data/ifeval_combined.json',
    'data/test_v4_enhanced.json',
]

for f in files:
    if not os.path.exists(f):
        print(f"{f}: NOT FOUND")
        continue
    d = json.load(open(f, 'r', encoding='utf-8'))
    keys = list(d[0].keys()) if d else []
    print(f"{f}: {len(d)} samples, keys={keys}")
    # Show first sample instruction prefix
    if d and 'instruction' in d[0]:
        print(f"  sample instr: {d[0]['instruction'][:80]}")
    if d and 'output' in d[0]:
        out = d[0]['output']
        print(f"  sample output: {out[:80] if out else '(empty)'}")
    print()
