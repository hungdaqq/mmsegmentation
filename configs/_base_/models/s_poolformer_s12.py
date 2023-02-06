# model settings
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s12_3rdparty_32xb128_in1k_20220414-f8d83051.pth'  # noqa
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='mmcls.PoolFormer',
        arch='s12',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file, prefix='backbone.'),
        in_patch_size=7,
        in_stride=4,
        in_pad=2,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.,
        drop_path_rate=0.,
        out_indices=(0, 2, 4, 6),
        frozen_stages=0,
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
