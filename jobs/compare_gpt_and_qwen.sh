#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."
echo -e "${COLOR_BLUE}当前工作目录已设置为: $(pwd)"

bash ./jobs/qwen3-14b.sh > "logs/qwen3-14b.log" 2>&1

bash ./jobs/gpt-oss-20b.sh > "logs/gpt-oss-20b.log" 2>&1

uv run -m utils.utils
