# A2Mamba for Semantic Segmentation

## 1. Requirements
We highly suggest using our provided dependencies to ensure reproducibility:
```
# Environments:
python==3.9
# Dependencies:
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install openmim
mim install mmcv-full==1.7.2
pip install mmsegmentation==0.30.0 mmdet==2.28.2 timm==0.6.12 einops natten
```
💡 Both the **A2Mamba backbone** and the **SegMAN-V2 decoder** require the `selective_scan_cuda_oflex_rh` CUDA extension, which can be built from the `selective_scan` folder of the [A2Mamba](https://github.com/LMMMEng/A2Mamba) repository.

💡 To enable torch>=2.1.0 to support mmcv 1.7.2, you need to make the following changes:

> 1️⃣ https://goo.su/XhU5vWr
> 2️⃣ https://goo.su/ogm4yO

## 2. Data Preparation
Prepare the datasets with the following folder structure:
```
/data/dataset/
├── ADEChallengeData2016/
│   ├── images/
│   └── annotations/
└── cityscapes/
    ├── leftImg8bit/
    └── gtFine/
```

## 3. Main Results on ADE20K using UperNet

| Model | Pretrained | Iters | mIoU (SS) | mIoU (MS) | config | Download |
|:-----:|:----------:|:-----:|:---------:|:---------:|:------:|:--------:|
| A2Mamba-S | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_small.pth) | 160K | 51.6 | 52.0 | [config](configs/upernet/upernet_a2mamba_small_ade20k_4xb4.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/upernet_a2mamba_small_ade20k.pth) |
| A2Mamba-B | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_base.pth) | 160K | 53.3 | 53.9 | [config](configs/upernet/upernet_a2mamba_base_ade20k_4xb4.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/tree/main/a2mamba_pretrained_weights) |
| A2Mamba-L | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_large.pth) | 160K | 53.7 | 54.1 | [config](configs/upernet/upernet_a2mamba_large_ade20k_4xb4.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/upernet_a2mamba_large_ade20k.pth) |

## 4. Main Results on Cityscapes using SegMAN-V2

| Model | Pretrained | mIoU | config | Download |
|:-----:|:----------:|:----:|:------:|:--------:|
| SegMAN-V2-N | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_nano.pth) | 81.0 | [config](configs/segmanv2/segmanv2_a2mamba_nano_cityscapes_4xb2.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/segmanv2_a2mamba_nano_cityscapes.pth) |
| SegMAN-V2-S | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_small.pth) | 83.8 | [config](configs/segmanv2/segmanv2_a2mamba_small_cityscapes_4xb2.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/segmanv2_a2mamba_small_cityscapes.pth) |
| SegMAN-V2-B | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_base.pth) | 84.2 | [config](configs/segmanv2/segmanv2_a2mamba_base_cityscapes_4xb2.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/segmanv2_a2mamba_base_cityscapes.pth) |
| SegMAN-V2-L | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_large.pth) | 84.6 | [config](configs/segmanv2/segmanv2_a2mamba_large_cityscapes_4xb2.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/segmanv2_a2mamba_large_cityscapes.pth) |

## 5. Train
To train A2Mamba with UperNet on ADE20K with 4 gpus, run:
```
bash scripts/dist_train.sh configs/upernet/upernet_a2mamba_small_ade20k_4xb4.py 4
```
To train A2Mamba with the SegMAN-V2 decoder on Cityscapes with 4 gpus, run:
```
bash scripts/dist_train.sh configs/segmanv2/segmanv2_a2mamba_small_cityscapes_4xb2.py 4
```

## 6. Validation
To evaluate A2Mamba with UperNet on ADE20K, run:
```
bash scripts/dist_test.sh configs/upernet/upernet_a2mamba_small_ade20k_4xb4.py path-to-checkpoint 4 --eval mIoU
```
To evaluate A2Mamba with the SegMAN-V2 decoder on Cityscapes, run:
```
bash scripts/dist_test.sh configs/segmanv2/segmanv2_a2mamba_small_cityscapes_4xb2.py path-to-checkpoint 4 --eval mIoU
```

## Citation
If you find this project useful for your research, please consider citing:
```
@article{lou2026a2mamba,
  title={A2Mamba: Attention-augmented State Space Models for Visual Recognition},
  author={Lou, Meng and Fu, Yunxiang and Yu, Yizhou},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026}
}
```
```
@inproceedings{fu2025segman,
  title={SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for Semantic Segmentation},
  author={Fu, Yunxiang and Lou, Meng and Yu, Yizhou},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19077--19087},
  year={2025},
}
```
