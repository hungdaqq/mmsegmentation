_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/cvc.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_150k.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.0005)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)