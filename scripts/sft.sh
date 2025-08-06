#!/bin/bash

cd ..
source .venv/bin/activate

# 设置环境变量
FINETUNED_DIR="/data/finetuning/finetuned-model"
MODEL_NAME="gpt-oss-20b"

export MODEL_ID="/data/download-model/${MODEL_NAME}"
export FINAL_ADAPTER_PATH="${FINETUNED_DIR}/${MODEL_NAME}-adapter"
export FINAL_MERGED_PATH="${FINETUNED_DIR}/${MODEL_NAME}-bid"
export EPOCHS=2
export LR=2e-5

CUDA_VISIBLE_DEVICES="0,1,6,7" \
  accelerate launch --config_file "finetuning/accelerate_config.yaml" \
    finetuning/sft.py
