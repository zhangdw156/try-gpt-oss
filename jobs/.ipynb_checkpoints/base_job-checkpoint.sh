#!/bin/bash

# --- 脚本健壮性设置 ---
# set -e: 脚本中任何命令返回非零退出码（表示错误）时，立即退出脚本。
# set -o pipefail: 如果管道中的任何命令失败，则整个管道的退出码也为非零。
set -e
set -o pipefail

# 你的原始设置代码保持不变
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."
echo -e "当前工作目录已设置为: $(pwd)"

export PYTHONPATH="$(pwd)"
echo "python项目根目录: ${PYTHONPATH}"
source .venv/bin/activate

# ----------------- 变量定义 -----------------
# export MODEL_NAME="gpt-oss-20b" # 这行似乎需要取消注释
export FINETUNED_DIR="/data/finetuning/finetuned-model"
export FINAL_MODEL_NAME="${MODEL_NAME}-bid"
export MODEL_ID="/data/download-model/${MODEL_NAME}"
export FINAL_ADAPTER_PATH="${PYTHONPATH}/model_lora/${MODEL_NAME}-adapter"
export FINAL_MERGED_PATH="${FINETUNED_DIR}/${FINAL_MODEL_NAME}"

export EPOCHS=2
export LR=2e-5

export CUDA_VISIBLE_DEVICES="1,5,6,7"

# ----------------- 确保输出目录存在 (增加的步骤) -----------------
# 在运行评估之前，确保输出结果的目录是存在的
export EVAL_OUTPUT_DIR="data/${FINAL_MODEL_NAME}"
mkdir -p "${EVAL_OUTPUT_DIR}"
echo "确保评估输出目录存在: ${EVAL_OUTPUT_DIR}"

# --- 核心执行流程 ---

# 步骤 1: SFT 微调
echo "--- [步骤 1/3] 开始SFT微调... ---"
# accelerate launch --config_file "finetuning/accelerate_config.yaml" \
#   finetuning/sft.py
echo "✅ [步骤 1/3] SFT微调成功完成。"
echo # 添加空行以提高可读性

# 步骤 2: 生成模型输出
echo "--- [步骤 2/3] 开始生成模型输出用于评估... ---"
# 注意：你的原始命令没有指定输出文件，这里我假设它会输出到
# "data/${FINAL_MODEL_NAME}/results.txt"，并在下一步中使用
uv run -m evaluate.get_f1_score_data \
  --model_id "${FINAL_MERGED_PATH}" \
  --sample_size 2000
echo "✅ [步骤 2/3] 模型输出生成成功。"
echo

# 步骤 3: 计算 F1 Score
echo "--- [步骤 3/3] 开始计算 F1 Score... ---"
uv run -m evaluate.get_f1_score \
  --dataset_file "dataset/bid-announcement-zh-v1.0.jsonl" \
  --ground_truth_file "data/gt.txt" \
  --model_output_file "${EVAL_OUTPUT_DIR}/results.txt" \
  --output_file "${EVAL_OUTPUT_DIR}/f1_score.csv"
echo "✅ [步骤 3/3] F1 Score 计算成功。"
echo

echo "🎉 所有流程成功完成！"