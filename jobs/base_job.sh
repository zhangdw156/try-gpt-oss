#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."
echo -e "${COLOR_BLUE}当前工作目录已设置为: $(pwd)"

export PYTHONPATH="$(pwd)"
echo "python项目根目录: ${PYTHONPATH}"
source .venv/bin/activate

# export MODEL_NAME="gpt-oss-20b"
export FINETUNED_DIR="/data/finetuning/finetuned-model"
export FINAL_MODEL_NAME="${MODEL_NAME}-bid"
export MODEL_ID="/data/download-model/${MODEL_NAME}"
export FINAL_ADAPTER_PATH="${PYTHONPATH}/model_lora/${MODEL_NAME}-adapter"
export FINAL_MERGED_PATH="${FINETUNED_DIR}/${FINAL_MODEL_NAME}"

export EPOCHS=2
export LR=2e-5

export CUDA_VISIBLE_DEVICES="3,5,6,7"

accelerate launch --config_file "finetuning/accelerate_config.yaml" \
  finetuning/sft.py

sleep 10

uv run -m evaluate.get_f1_score_data \
  --model_id "${FINAL_MERGED_PATH}" \
  --sample_size 2000

sleep 10

uv run -m evaluate.get_f1_score \
  --dataset_file "dataset/bid-announcement-zh-v1.0.jsonl" \
  --ground_truth_file "data/gt.txt" \
  --model_output_file "data/${FINAL_MODEL_NAME}/results.txt" \
  --output_file "data/${FINAL_MODEL_NAME}/f1_score.csv"