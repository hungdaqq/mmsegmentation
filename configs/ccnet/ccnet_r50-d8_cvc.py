_base_ = [
    '../_base_/models/ccnet_r50-d8.py',
    '../_base_/datasets/cvc.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_150k.py'
]
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
