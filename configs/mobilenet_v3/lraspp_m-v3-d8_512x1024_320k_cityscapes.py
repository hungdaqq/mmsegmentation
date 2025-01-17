_base_ = [
    '../_base_/models/lraspp_m-v3-d8.py', '../_base_/datasets/skin.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_150k.py'
]

model = dict(pretrained='open-mmlab://contrib/mobilenet_v3_large')

# Re-config the data sampler.
data = dict(samples_per_gpu=4, workers_per_gpu=4)