_base_ = [
    'mmdet::_base_/default_runtime.py',
]
load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"

model = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='MultichannelDetDataPreprocessor',
        pad_size_divisor=32,
        mean=(0, 0, 0),
        std=(4081.0, 4081.0, 4081.0)
    ),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.375,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        preproc_transform=dict(type="ram", functions=['wb', 'ccm', 'gamma', 'brightness'],
                               ffm_params=dict(ffm_type='BN_HG'), clamp_values=True),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=3,
        in_channels=96,
        feat_channels=96,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

max_epochs = 300
num_last_epochs = 15
interval = 5

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
# optimizer
# default 8 gpu
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


default_hooks = dict(
    checkpoint=dict(max_keep_ckpts=3, interval=5, save_best=['coco/bbox_mAP']),
    logger=dict(type='LoggerHook', interval=50),
)

custom_hooks = [
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

train_pipeline = [
    dict(type='LoadNumpyFromFile'),
    dict(type='RGGBtoRGB'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize', scale=[(602, 320), (602, 400)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadNumpyFromFile'),
    dict(type='RGGBtoRGB'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(602, 400), keep_ratio=True),

    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# dataset settings
dataset_type = 'PascalRawDataset'
data_root = '/data/pascalraw/pascalraw_h400/'

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/dng_instances_train.json',
        data_prefix=dict(img='dng/'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/dng_instances_test.json',
        data_prefix=dict(img='dng/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/dng_instances_test.json',
    metric='bbox',
    format_only=False, classwise=True)
test_evaluator = val_evaluator
