MODEL=a2mamba_t # a2mamba_{n, t, s, b, l}
python3 validate.py \
/data/dataset/imagenet \
--model $MODEL -b 128 \
--pretrained # or --checkpoint /path/to/checkpoint 