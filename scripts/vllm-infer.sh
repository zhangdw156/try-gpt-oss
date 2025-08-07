#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."
echo -e "当前工作目录已设置为: $(pwd)"

export PYTHONPATH="$(pwd)"
echo "python项目根目录: ${PYTHONPATH}"

source .venv/bin/activate

export MODEL_NAME="gpt-oss-20b"

export FINETUNED_DIR="/data/finetuning/finetuned-model"
export FINAL_MODEL_NAME="${MODEL_NAME}-bid"
export FINAL_MERGED_PATH="${FINETUNED_DIR}/${FINAL_MODEL_NAME}"

CUDA_VISIBLE_DEVICES="0,1,6,7" \
uv run -m evaluate.get_f1_score_data_with_vllm \
  --model_id "${FINAL_MERGED_PATH}" \
  --dataset_id "./dataset/bid-announcement-zh-v1.0.jsonl" \
  --sample_size 200

uv run -m evaluate.get_f1_score \
  --dataset_file "./dataset/bid-announcement-zh-v1.0.jsonl" \
  --ground_truth_file "data/gt.txt" \
  --model_output_file "data/${FINAL_MODEL_NAME}/results.txt" \
  --output_file "./data/${FINAL_MODEL_NAME}/f1_score.csv"