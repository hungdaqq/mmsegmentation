_base_ = [
    '../_base_/models/FCB-MiTb0.py', '../_base_/datasets/bowl.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_150k.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

model = dict(
    # pretrained=checkpoint,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint), embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 6, 3]),
    neck = dict(in_channels=[64, 128, 320, 512], out_channels=256),
    decode_head=dict(in_channels=[256, 256, 256, 256], feature_strides=[4, 8, 16, 32], channels=128, num_classes=2, out_channels=2,
    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
    )
    
# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=4)
