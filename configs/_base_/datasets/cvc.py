# dataset settings
dataset_type = 'CVCDataset'
data_root = '/data/CVC-Clinic/'
img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
img_scale = (224, 224)
crop_size = (200, 200)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=40000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/images',
            ann_dir='train/masks_gray',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/Kvasir/images',
        ann_dir='test/Kvasir/masks_gray',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='test/ETIS-LaribPolypDB/images',
        # ann_dir='test/ETIS-LaribPolypDB/masks_gray',
        # img_dir='test/CVC-ClinicDB/images',
        # ann_dir='test/CVC-ClinicDB/masks_gray',
        # img_dir='test/Kvasir/images',
        # ann_dir='test/Kvasir/masks_gray',
        img_dir='test/CVC-300/images',
        ann_dir='test/CVC-300/masks_gray',
        pipeline=test_pipeline))
