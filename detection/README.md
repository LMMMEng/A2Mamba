# Applying A2Mamba to Object Detection and Instance Segmentation

## 1. Requirements

```
pip install mmcv-full==1.7.2 --no-cache-dir
pip install mmdet==2.28.2 --no-cache-dir
```
> 💡 The A2Mamba backbone requires the `selective_scan_cuda_oflex_rh` CUDA extension,
> which can be built from the `selective_scan` folder of the [A2Mamba](https://github.com/LMMMEng/A2Mamba) repository.

💡 To enable torch>=2.1.0 to support mmcv 1.7.2, you need to make the following changes:

> 1️⃣ https://goo.su/XhU5vWr
> 2️⃣ https://goo.su/ogm4yO

## 2. Data Preparation

Prepare COCO 2017 according to the [guidelines](https://github.com/open-mmlab/mmdetection/blob/2.x/docs/en/1_exist_data_model.md).

## 3. Main Results on COCO using Mask R-CNN framework

| Backbone |                                                    Pretrain                                                    | Schedule | AP_b | AP_m |                                Config                                |                                                         Download                                                         |
| :-------: | :-------------------------------------------------------------------------------------------------------------: | :------: | :--: | :--: | :-------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
| A2Mamba-S | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_small.pth) |    3x    | 51.5 | 45.3 | [config](configs/maskrcnn/mask_rcnn_a2mamba_small_in1k_fpn_3x_coco.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/maskrcnn3x_a2mamba_small_coco.pth) |
| A2Mamba-B | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_base.pth) |    3x    | 52.7 | 46.8 | [config](configs/maskrcnn/mask_rcnn_a2mamba_base_in1k_fpn_3x_coco.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/maskrcnn3x_a2mamba_base_coco.pth) |
| A2Mamba-L | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_large.pth) |    3x    | 53.0 | 46.8 | [config](configs/maskrcnn/mask_rcnn_a2mamba_large_in1k_fpn_3x_coco.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/maskrcnn3x_a2mamba_large_coco.pth) |

## 4. Main Results on COCO using Cascade Mask R-CNN framework

| Backbone |                                                    Pretrain                                                    | Schedule | AP_b | AP_m |                                        Config                                        |                                                             Download                                                             |
| :-------: | :-------------------------------------------------------------------------------------------------------------: | :------: | :--: | :--: | :-----------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------: |
| A2Mamba-S | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_small.pth) |    3x    | 54.0 | 46.6 | [config](configs/cascade_maskrcnn/cascade_mask_rcnn_a2mamba_small_in1k_fpn_3x_coco.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/cascademaskrcnn3x_a2mamba_small_coco.pth) |
| A2Mamba-B | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_base.pth) |    3x    | 55.4 | 47.6 | [config](configs/cascade_maskrcnn/cascade_mask_rcnn_a2mamba_base_in1k_fpn_3x_coco.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/cascademaskrcnn3x_a2mamba_base_coco.pth) |
| A2Mamba-L | [ImageNet-1K](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_large.pth) |    3x    | 55.6 | 48.1 | [config](configs/cascade_maskrcnn/cascade_mask_rcnn_a2mamba_large_in1k_fpn_3x_coco.py) | [model](https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/cascademaskrcnn3x_a2mamba_large_coco.pth) |

## 5. Train

To train ``A2Mamba-S + Mask R-CNN 3x`` model on COCO dataset with 8 GPUs (single node), run:

```
NUM_GPUS=8
CONFIG=configs/maskrcnn/mask_rcnn_a2mamba_small_in1k_fpn_3x_coco.py
bash scripts/dist_train.sh $CONFIG $NUM_GPUS
```

> 💡If your GPU memory is insufficient, you can enable gradient checkpointing by adding ``use_checkpoint=[2, 0, 0, 0]`` to the ``backbone`` dict in the config. If you're still experiencing memory issues, you can increase these values, but be aware that this may slow down training.

## 6. Validation

To evaluate ``A2Mamba-S + Mask R-CNN 3x`` model on COCO dataset, run:

```
NUM_GPUS=8
CKPT=path-to-checkpoint.pth
CONFIG=configs/maskrcnn/mask_rcnn_a2mamba_small_in1k_fpn_3x_coco.py
bash scripts/dist_test.sh $CONFIG $CKPT $NUM_GPUS --eval bbox segm
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
