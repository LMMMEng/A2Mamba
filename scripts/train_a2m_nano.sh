#!/usr/bin/env bash
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
python3 -m torch.distributed.launch \
--master_port=$((RANDOM+10000)) \
--nproc_per_node=8 \
train.py \
--data-dir /data/dataset/imagenet/ \
--batch-size 256 \
--model a2mamba_n \
--lr 4e-3 \
--drop-path 0 \
--epochs 300 \
--warmup-epochs 5 \
--workers 8 \
--output output/a2mamba_n/ \
--native-amp \
--clip-grad 5 \
--mixup 0.1 \
--color-jitter 0