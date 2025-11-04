# [A2Mamba: Attention-augmented State Space Models for Visual Recognition](https://arxiv.org/abs/2507.16624)

This is an official PyTorch implementation of "[**A2Mamba: Attention-augmented State Space Models for Visual Recognition**](https://arxiv.org/abs/2507.16624)".

# Introduction
Transformers and Mamba, initially invented for natural language processing, have inspired backbone architectures for visual recognition. Recent studies integrated Local Attention Transformers with Mamba to capture both local details and global contexts. Despite competitive performance, these methods are limited to simple stacking of Transformer and Mamba layers without any interaction mechanism between them. Thus, deep integration between Transformer and Mamba layers remains an open problem. We address this problem by proposing A2Mamba, a powerful Transformer-Mamba hybrid network architecture, featuring a new token mixer termed Multi-scale Attention-augmented State Space Model (MASS), where multi-scale attention maps are integrated into an attention-augmented SSM (A2SSM). A key step of A2SSM performs a variant of cross-attention by spatially aggregating the SSM's hidden states using the multi-scale attention maps, which enhances spatial dependencies pertaining to a two-dimensional space while improving the dynamic modeling capabilities of SSMs. Our A2Mamba outperforms all previous ConvNet-, Transformer-, and Mamba-based architectures in visual recognition tasks. For instance, A2Mamba-L achieves an impressive 86.1% top-1 accuracy on ImageNet-1K. In semantic segmentation, A2Mamba-B exceeds CAFormer-S36 by 2.5% in mIoU, while exhibiting higher efficiency. In object detection and instance segmentation with Cascade Mask R-CNN, A2Mamba-S surpasses MambaVision-B by 1.2%/0.9% in AP^b/AP^m, while having 40% less parameters.
<center> 
<img src="images/img.jpg" width="60%" height="auto">
</center>

# Image Classification

## 1. Requirements
We highly suggest using our provided dependencies to ensure reproducibility:
```
# Environments:
cuda==12.1
python==3.10
# Packages:
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels/
pip install timm==0.6.12
pip install mmengine==0.2.0
# Other dependencies:
cd selective_scan; pip install .
```

>💡 If you encounter network issues during the installation of ``natten``, please download this [**package**](https://github.com/LMMMEng/OverLoCK/releases/download/v1/natten-0.17.1+torch230cu121-cp310-cp310-linux_x86_64.whl) and install it locally.


## 2. Data Preparation
Prepare [ImageNet](https://image-net.org/) with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## 3. Main Results on ImageNet-1K with Pretrained Models

| Models        | Input Size | FLOPs (G) | Params (M) | Top-1 (%) | Download |
|:-------------:|:----------:|:---------:|:----------:|:---------:|:--------:|
| A2Mamba-Nano  | 224x224   | 0.8       | 4          | 78.7      | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_nano.pth) |
| A2Mamba-Tiny  | 224x224   | 2.7       | 15         | 83.0      | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_tiny.pth) |
| A2Mamba-Small | 224x224   | 5.4       | 31         | 84.7      | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_small.pth) |
| A2Mamba-Base  | 224x224   | 10.7      | 51         | 85.7      | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_base.pth) |
| A2Mamba-Large | 224x224   | 17.4      | 95         | 86.1      | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_large.pth) |

## 4. Train
To train ```A2Mamba``` models on ImageNet-1K with 8 gpus (single node), run:
```
sh scripts/train_a2m_nano.sh # train A2Mamba-Nano
sh scripts/train_a2m_tiny.sh # train A2Mamba-Tiny
sh scripts/train_a2m_small.sh # train A2Mamba-Small
sh scripts/train_a2m_base.sh # train A2Mamba-Base
sh scripts/train_a2m_large.sh # train A2Mamba-Large
```  
> 💡If you encounter NaN loss, please delete ``--native-amp`` to disable AMP training and resume the checkpoint before the NaN loss occurred.
>   
> 💡If your GPU memory is insufficient, you can enable gradient checkpointing by adding the following arguments: ``--grad-checkpoint --ckpt-stg 4 0 0 0``. If you're still experiencing memory issues, you can increase these values, but be aware that this may slow down training.

## 5. Validation
To evaluate ```A2Mamba``` on ImageNet-1K, run:
```
MODEL=a2mamba_t # a2mamba_{n, t, s, b, l}
python3 validate.py \
/path/to/imagenet \
--model $MODEL -b 128 \
--pretrained # or --checkpoint /path/to/checkpoint 
```

# Citation
If you find this project useful for your research, please consider citing:
```
@article{lou2025a2mamba,
  title={A2Mamba: Attention-augmented State Space Models for Visual Recognition},
  author={Lou, Meng and Fu, Yunxiang and Yu, Yizhou},
  journal={arXiv preprint arXiv:2507.16624},
  year={2025}
}
```

# Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.  
> [timm](https://github.com/rwightman/pytorch-image-models), [natten](https://github.com/SHI-Labs/NATTEN), [mmcv](https://github.com/open-mmlab/mmcv), [vmamba](https://github.com/MzeroMiko/VMamba) 

# Contact
If you have any questions, please feel free to [open an issue](https://github.com/LMMMEng/A2Mamba/issues/new) or [contact me](lmzmm.0921@gmail.com).
