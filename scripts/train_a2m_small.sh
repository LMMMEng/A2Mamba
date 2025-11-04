#!/usr/bin/env bash
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
python3 -m torch.distributed.launch \
--master_port=$((RANDOM+10000)) \
--nproc_per_node=8 \
train.py \
--data-dir /data/dataset/imagenet/ \
--batch-size 256 \
--model a2mamba_s \
--lr 1e-3 \
--auto-lr \
--drop-path 0.2 \
--epochs 300 \
--warmup-epochs 5 \
--workers 8 \
--output output/a2mamba_s/ \
--mesa 1 \
--model-ema \
--model-ema-decay 0.9999 \
--native-amp \
--clip-grad 5