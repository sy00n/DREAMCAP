img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
# model settings
evidence_loss = dict(type='EvidenceLoss',
                      num_classes=48,
                      evidence='exp',
                      loss_type='log',
                      with_kldiv=False,
                      with_avuloss=True,
                      annealing_method='exp')
train_cfg = None
test_cfg = dict(average_clips='evidence', evidence_type='exp')
model = dict(
    type='DREAMRecognizer3D',
    backbone=dict(
        type='DREAM3dSlowFast',
        pretrained=None,
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        rgb_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            in_channels=3,
            norm_eval=False),
        ske_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            in_channels=17,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        loss_cls=evidence_loss,
        in_channels=2304,  # 2048+256
        num_classes=48,
        spatial_type='avg',
        dropout_ratio=0.5),
    debias_head=dict(
        type='DebiasHead',
        loss_cls=evidence_loss,  # actually not used!
        loss_factor=0.1,
        num_classes=48,
        in_channels=2048,  # only slow features are debiased
        dropout_ratio=0.5,
        init_std=0.01))

dataset_type = {'rgb':'VideoDataset',
                'skeleton':'PoseDataset'}
ann_file = {
    "rgb":'../data/diving48/new_diving48_rgb.pkl',
    "skeleton":'../data/diving48/new_diving48_ske.pkl'
}
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = {
    "rgb":[
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
    ],
    "skeleton":[
        dict(type='UniformSampleFrames', clip_len=32),
        dict(type='PoseDecode'),
        dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
        dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
        dict(type='FormatShape', input_format='NCTHW_Heatmap'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]
}
val_pipeline = {
    'rgb':[
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
    ],
    'skeleton':[
        dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
        dict(type='PoseDecode'),
        dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
        dict(type='Resize', scale=(64, 64), keep_ratio=False),
        dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
        dict(type='FormatShape', input_format='NCTHW_Heatmap'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
}
test_pipeline = {
    'rgb':[
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
    ],
    'skeleton':[
        dict(type='UniformSampleFrames', clip_len=48, num_clips=10, test_mode=True),
        dict(type='PoseDecode'),
        dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
        dict(type='Resize', scale=(64, 64), keep_ratio=False),
        dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp),
        dict(type='FormatShape', input_format='NCTHW_Heatmap'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
}
dual_modality = True
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=1,
    test_dataloader=dict(videos_per_gpu=1),
    train={
        "rgb":dict(
            type='RepeatDataset',
            times=10,
            dataset=dict(type=dataset_type['rgb'], ann_file=ann_file['rgb'], split='train', data_prefix='../data/diving48', pipeline=train_pipeline['rgb'])),
        "skeleton":dict(
            type='RepeatDataset',
            times=10,
            dataset=dict(type=dataset_type['skeleton'], ann_file=ann_file['skeleton'], split='train', pipeline=train_pipeline['skeleton'])),
    },
    val={
        "rgb":dict(type=dataset_type['rgb'], ann_file=ann_file['rgb'], split='test', data_prefix='../data/diving48', pipeline=val_pipeline['rgb']),
        "skeleton":dict(type=dataset_type['skeleton'], ann_file=ann_file['skeleton'], split='test', pipeline=val_pipeline['skeleton']),
    },
    test={
        "rgb":dict(type=dataset_type['rgb'], ann_file=ann_file['rgb'], split='test', data_prefix='../data/diving48', pipeline=test_pipeline['rgb']),
        "skeleton":dict(type=dataset_type['skeleton'], ann_file=ann_file['skeleton'], split='test', pipeline=test_pipeline['rgb'])
    }
)
# optimizer
optimizer = dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5)
total_epochs = 50
checkpoint_config = dict(interval=10)
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