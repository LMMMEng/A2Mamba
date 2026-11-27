#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=$((RANDOM+10000))

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$PORT train.py $CONFIG --launcher pytorch ${@:3}