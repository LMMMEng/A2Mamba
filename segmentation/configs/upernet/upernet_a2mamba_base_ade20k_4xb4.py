_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]


embed_dims = [96, 192, 384, 512]
model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='a2mamba_b',
        pretrained=True,
        drop_path_rate=0.4
    ),
    decode_head=dict(
        in_index=[0, 1, 2, 3],
        in_channels=embed_dims,
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_index=2,
        in_channels=embed_dims[2],
        num_classes=150
    ),
)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True,
                 policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0,
                 min_lr=0.0,
                 by_epoch=False)


data = dict(samples_per_gpu=4) # as gpus = 4
checkpoint_config = dict(interval=8000, max_keep_ckpts=1)
evaluation = dict(interval=8000, pre_eval=True, save_best='mIoU')

# place holder for new verison mmseg compatiability
resume_from = None
device = 'cuda'

# AMP (faster but may meet nan loss) ->
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 = dict()
