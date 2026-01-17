#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# support tp1 train eagle3 for qwen2.5-vl-7b-instruct
NUM_GPUS=8
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
<<<<<<< HEAD
    --target-model-path /disk3/wjp/pretrained_models/Qwen2.5-VL-7B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen2.5-vl-7b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/allava4v_train_2w.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/Qwen2.5-VL-7B-Instruct-2w \
=======
    --target-model-path /home/qspace/Qwen2.5-VL-7B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen2-5-vl-7b-eagle3.json \
    --train-data-path /mnt/cephfs/user_xuanweifu/data/datasets/ALLaVA/processed/allava4v_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/Qwen2.5-VL-7B-Instruct \
>>>>>>> 99146a9 (format code)
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --dist-timeout 360 \
    --chat-template qwen2-vl \
    --target-model-backend sglang \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
<<<<<<< HEAD
    --tp-size 1 \
    --sglang-mem-fraction-static 0.7 \
    --is-vlm \
    --min-pixels 200704 \
    --max-pixels 1003520 \
    --report-to tensorboard
=======
    --tp-size 4 \
    --sglang-mem-fraction-static 0.5 \
    --is-vlm \
    --min-pixels 50176 \
    --max-pixels 802816 \
    --report-to wandb \
    --wandb-key local-3a5edad7b716f6135697b662e2716e785ba80432 \
    --wandb-project luban-eagle3 \
    --wandb-name Qwen2.5-VL-7B-Instruct-allava4v
>>>>>>> 99146a9 (format code)
