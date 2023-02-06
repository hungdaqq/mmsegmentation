_base_ = [
    '../_base_/models/pspnet_unet_s5-d16.py', '../_base_/datasets/cvc2.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_150k.py'
]
model = dict(test_cfg=dict(crop_size=(128, 128), stride=(170, 170)))

