#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=$((RANDOM+10000))

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$PORT test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}