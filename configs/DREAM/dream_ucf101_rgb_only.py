img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
# model settings
evidence_loss = dict(type='EvidenceLoss',
                      num_classes=101,
                      evidence='exp',
                      loss_type='log',
                      with_kldiv=False,
                      with_avuloss=True,
                      annealing_method='exp')
train_cfg = None
test_cfg = dict(average_clips='evidence', evidence_type='exp')
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        lateral=False,
        conv1_kernel=(1, 7, 7),
        dilations=(1, 1, 1, 1),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        in_channels=3,
        norm_eval=False),
    cls_head=dict(
        type='I3DHead',
        loss_cls=evidence_loss,
        in_channels=2048,  # 2048+256
        num_classes=101,
        spatial_type='avg',
        dropout_ratio=0.5),
    debias_head=dict(
        type='DebiasHead',
        loss_cls=evidence_loss,  # actually not used!
        loss_factor=0.1,
        num_classes=101,
        in_channels=2048,  # only slow features are debiased
        dropout_ratio=0.5,
        init_std=0.01))

dataset_type = 'VideoDataset'
ann_file = '../data/ucf101/ucf101_dual.pkl'
train_pipeline = [
        dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='RandomResizedCrop'),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]
val_pipeline = [
        dict(
            type='SampleFrames',
            clip_len=32,
            frame_interval=2,
            num_clips=1,
            test_mode=True),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Flip', flip_ratio=0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
test_pipeline = [
        dict(
            type='SampleFrames',
            clip_len=32,
            frame_interval=2,
            num_clips=10,
            test_mode=True),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='ThreeCrop', crop_size=256),
        dict(type='Flip', flip_ratio=0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
dual_modality = False
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=0,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
            type='RepeatDataset',
            times=10,
            dataset=dict(type=dataset_type, ann_file=ann_file, split='train', data_prefix='../data/ucf101/rawframes', pipeline=train_pipeline, start_index=1)),
    val=dict(type=dataset_type, ann_file=ann_file, split='test', data_prefix='../data/ucf101/rawframes', pipeline=val_pipeline, start_index=1),
    test=dict(type=dataset_type, ann_file=ann_file, split='test', data_prefix='../data/ucf101/rawframes', pipeline=test_pipeline, start_index=1)
)
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup=None,
    warmup_by_epoch=True,
    warmup_iters=34)
total_epochs = 50
checkpoint_config = dict(interval=10, by_epoch=False)
workflow = [('train', 1)]
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
annealing_runner = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './output/dreamnet'
resume_from = None
load_from = None
find_unused_parameters = False