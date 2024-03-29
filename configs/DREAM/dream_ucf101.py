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
    type='DREAMRecognizer3D',
    backbone=dict(
        type='DREAM',
        pretrained=None,
        rgb_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            base_channels=64,
            fusion_kernel=4,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            in_channels=3,
            norm_eval=False,
            lateral_channels=32, # Must same base channel with ske_pathway!
            lateral_num_stages=3, # Must same num_stages with ske_pathway!
            lateral_lambda=1 # Weight to indicate how much skeleton modality will be reflected
            ),
        ske_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=32,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1),
            conv1_stride=(1, 1),
            pool1_stride=(1, 1),
            inflate=(0, 1, 1),
            in_channels=17,
            norm_eval=False,
            num_stages=3,
            stage_blocks=(3,4,6),
            out_indices=(2,),
            spatial_strides=(1, 2, 2),
            temporal_strides=(1, 1, 1))),
    cls_head=dict(
        type='SlowFastHead',
        loss_cls=evidence_loss,
        in_channels=2560,  # RGB output channel + SKE output channel
        num_classes=101,
        spatial_type='avg',
        dropout_ratio=0.5))

dataset_type = {'rgb':'VideoDataset',
                'skeleton':'PoseDataset'}

ann_file = '../data/ucf101/ucf101_dual.pkl'
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
    workers_per_gpu=0,
    test_dataloader=dict(videos_per_gpu=1),
    train={
        "rgb":dict(
            type='RepeatDataset',
            times=10,
            dataset=dict(type=dataset_type['rgb'], ann_file=ann_file, split='train', data_prefix='../data/ucf101/rawframes', pipeline=train_pipeline['rgb'], start_index=1)),
        "skeleton":dict(
            type='RepeatDataset',
            times=10,
            dataset=dict(type=dataset_type['skeleton'], ann_file=ann_file, split='train', pipeline=train_pipeline['skeleton'])),
    },
    val={
        "rgb":dict(type=dataset_type['rgb'], ann_file=ann_file, split='test', data_prefix='../data/ucf101/rawframes', pipeline=val_pipeline['rgb'], start_index=1),
        "skeleton":dict(type=dataset_type['skeleton'], ann_file=ann_file, split='test', pipeline=val_pipeline['skeleton']),
    },
    test={
        "rgb":dict(type=dataset_type['rgb'], ann_file=ann_file, split='test', data_prefix='../data/ucf101/rawframes', pipeline=test_pipeline['rgb'], start_index=1),
        "skeleton":dict(type=dataset_type['skeleton'], ann_file=ann_file, split='test', pipeline=test_pipeline['rgb'])
    }
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
load_from = {
                "rgb_path":"https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth",
                "ske_path":"https://download.openmmlab.com/mmaction/skeleton/posec3d/k400_posec3d-041f49c6.pth"
             }
find_unused_parameters = False