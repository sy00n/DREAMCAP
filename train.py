import argparse
import json
from collections import namedtuple
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmaction import __version__
from mmaction.apis import train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.cfg, 'r') as f:
        args = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    cfg = Config.fromfile(args.config)

    try:
        cfg.merge_from_dict(args.cfg_options)
    except:
        pass

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    
    if cfg.dual_modality:
        if cfg.omnisource:
            # If omnisource flag is set, cfg.data.train should be a list
            assert type(cfg.data.train) is list
            rgb_datasets = [build_dataset(dataset) for dataset in cfg.data.train.rgb]
            ske_datasets = [build_dataset(dataset) for dataset in cfg.data.train.skeleton]
        else:
            rgb_datasets = [build_dataset(cfg.data.train.rgb)]
            ske_datasets = [build_dataset(cfg.data.train.skeleton)]

        if len(cfg.workflow) == 2:
            # For simplicity, omnisource is not compatiable with val workflow,
            # we recommend you to use `--validate`
            assert not cfg.omnisource
            if args.validate:
                warnings.warn('val workflow is duplicated with `--validate`, '
                            'it is recommended to use `--validate`. see '
                            'https://github.com/open-mmlab/mmaction2/pull/123')
            rgb_val_dataset = copy.deepcopy(cfg.data.val.rgb)
            ske_val_dataset = copy.deepcopy(cfg.data.val.skeleton)
            rgb_datasets.append(build_dataset(rgb_val_dataset))
            ske_datasets.append(build_dataset(ske_val_dataset))
        
        datasets = [rgb_datasets, ske_datasets]
    else:
        if cfg.omnisource:
            # If omnisource flag is set, cfg.data.train should be a list
            assert type(cfg.data.train) is list
            datasets = [build_dataset(dataset) for dataset in cfg.data.train]
        else:
            datasets = [build_dataset(cfg.data.train)]

        if len(cfg.workflow) == 2:
            # For simplicity, omnisource is not compatiable with val workflow,
            # we recommend you to use `--validate`
            assert not cfg.omnisource
            if args.validate:
                warnings.warn('val workflow is duplicated with `--validate`, '
                            'it is recommended to use `--validate`. see '
                            'https://github.com/open-mmlab/mmaction2/pull/123')
            val_dataset = copy.deepcopy(cfg.data.val)
            datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__ + get_git_hash(digits=7),
            config=cfg.text)

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()