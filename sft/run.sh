#!/bin/bash

source ../.venv/bin/activate

CUDA_VISIBLE_DEVICES="0,1,6,7" \
accelerate launch --config_file "accelerate_config.yaml" \
main.py