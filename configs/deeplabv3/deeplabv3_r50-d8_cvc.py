_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/cvc.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_150k.py'
]
model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))
optimizer = dict(type='SGD', lr=0.0004, momentum=0.9, weight_decay=0.0001)
