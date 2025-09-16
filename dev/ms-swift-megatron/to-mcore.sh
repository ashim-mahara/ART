#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 \
swift export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir Qwen3-30B-A3B-Instruct-2507-mcore \
    --test_convert_precision true
    