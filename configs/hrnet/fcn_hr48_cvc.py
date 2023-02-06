_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/cvc.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_150k.py'
]
# model = dict(
#     decode_head=dict(num_classes=2),
#     test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)