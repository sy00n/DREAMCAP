import copy as cp
from itertools import chain

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer)
from mmcv.runner.hooks import Fp16OptimizerHook
from mmcv.runner.checkpoint import load_url_dist

from ..core import (DistEpochEvalHook, EpochEvalHook,
                    OmniSourceDistSamplerSeedHook, OmniSourceRunner, AnnealingRunner,
                    DualOmniSourceRunner, DualAnnealingRunner, DualEpochBasedRunner)
from ..datasets import build_dataloader, build_dataset
from ..utils import get_root_logger

class MMDataParallel_(MMDataParallel):
    def train_step(self, *inputs, **kwargs):
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.train_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             'instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        
        return self.module.train_step(*inputs[0], **kwargs[0])
    
    def val_step(self, *inputs, **kwargs):
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.val_step(*inputs, **kwargs[0])

        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.val_step(*inputs[0], **kwargs[0])


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.log_level)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare data loaders
    
    if cfg.dual_modality:
        rgb_dataset = dataset[0] if isinstance(dataset[0], (list, tuple)) else [dataset[0]]
        ske_dataset = dataset[1] if isinstance(dataset[1], (list, tuple)) else [dataset[1]]
        from ..datasets import DREAMDataset
        dataset = [DREAMDataset(rgb_dataset[0], ske_dataset[0])]
    else:
        dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        pin_memory=cfg.data.get('pin_memory', True))  # by default, pin_memory=True
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    if cfg.omnisource:
        # The option can override videos_per_gpu
        train_ratio = cfg.data.get('train_ratio', [1] * len(dataset))
        omni_videos_per_gpu = cfg.data.get('omni_videos_per_gpu', None)
        if omni_videos_per_gpu is None:
            dataloader_settings = [dataloader_setting] * len(dataset)
        else:
            dataloader_settings = []
            for videos_per_gpu in omni_videos_per_gpu:
                this_setting = cp.deepcopy(dataloader_setting)
                this_setting['videos_per_gpu'] = videos_per_gpu
                dataloader_settings.append(this_setting)
        data_loaders = [
            build_dataloader(ds, **setting)
            for ds, setting in zip(dataset, dataloader_settings)
        ]

    else:
        data_loaders = [
            build_dataloader(ds, **dataloader_setting) for ds in dataset
        ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel_(
            model.to(device))

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    # Runner = OmniSourceRunner if cfg.omnisource else EpochBasedRunner
    if cfg.dual_modality:
        if cfg.omnisource:
            Runner = DualOmniSourceRunner
        elif cfg.get('annealing_runner', False):
            Runner = DualAnnealingRunner  # add annealing runner support
        else:
            Runner = DualEpochBasedRunner
    else:
        if cfg.omnisource:
            Runner = OmniSourceRunner
        elif cfg.get('annealing_runner', False):
            Runner = AnnealingRunner  # add annealing runner support
        else:
            Runner = EpochBasedRunner
    runner = Runner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if cfg.omnisource:
            runner.register_hook(OmniSourceDistSamplerSeedHook())
        else:
            runner.register_hook(DistSamplerSeedHook())

    if validate:
        if cfg.dual_modality:
            eval_cfg = cfg.get('evaluation', {})
            rgb_val_dataset = build_dataset(cfg.data.val.rgb, dict(test_mode=True))
            ske_val_dataset = build_dataset(cfg.data.val.skeleton, dict(test_mode=True))
            dataloader_setting = dict(
                videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
                workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
                # cfg.gpus will be ignored if distributed
                num_gpus=len(cfg.gpu_ids),
                dist=distributed,
                pin_memory=cfg.data.get('pin_memory', True),  # by default, pin_memory=True
                shuffle=False)
            dataloader_setting = dict(dataloader_setting,
                                    **cfg.data.get('val_dataloader', {}))
            rgb_val_dataloader = build_dataloader(rgb_val_dataset, **dataloader_setting)
            ske_val_dataloader = build_dataloader(ske_val_dataset, **dataloader_setting)
            val_dataloader = zip(rgb_val_dataloader, ske_val_dataloader)
            eval_hook = DistEpochEvalHook if distributed else EpochEvalHook
            #runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
        else:
            eval_cfg = cfg.get('evaluation', {})
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            dataloader_setting = dict(
                videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
                workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
                # cfg.gpus will be ignored if distributed
                num_gpus=len(cfg.gpu_ids),
                dist=distributed,
                pin_memory=cfg.data.get('pin_memory', True),  # by default, pin_memory=True
                shuffle=False)
            dataloader_setting = dict(dataloader_setting,
                                    **cfg.data.get('val_dataloader', {}))
            val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
            eval_hook = DistEpochEvalHook if distributed else EpochEvalHook
            #runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        if hasattr(cfg.load_from, dict):
            for description, url in cfg.load_from.items():
                x = load_url_dist(url)
                dct = {}
                for key, value in x["state_dict"].items():
                    key_ = key.split(".")
                    key_.insert(1, description)
                    dct.update({".".join(key_):value})
                runner.load_checkpoint(dct, strict=False)
        else:
            runner.load_checkpoint(cfg.load_from)
    runner_kwargs = dict()
    if cfg.omnisource:
        runner_kwargs = dict(train_ratio=train_ratio)
    if cfg.get('annealing_runner', False):
        runner_kwargs.update(annealing=True)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, **runner_kwargs)
