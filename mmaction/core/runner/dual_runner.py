import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from mmcv.runner import EpochBasedRunner, Hook
from mmcv.runner.utils import get_host_info

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

class DualEpochBasedRunner(EpochBasedRunner):
    def run_iter(self, rgb_data_batch, ske_data_batch, train_mode, **kwargs):
        
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, rgb_data_batch, ske_data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(rgb_data_batch, ske_data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(rgb_data_batch, ske_data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, (rgb_data_batch, ske_data_batch) in enumerate(self.data_loader):
            
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(rgb_data_batch, ske_data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, (rgb_data_batch, ske_data_batch) in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                self.run_iter(rgb_data_batch, ske_data_batch, train_mode=False, **kwargs)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

class OmniSourceDistSamplerSeedHook(Hook):

    def before_epoch(self, runner):
        for data_loader in runner.data_loaders:
            if hasattr(data_loader.sampler, 'set_epoch'):
                # in case the data loader uses `SequentialSampler` in Pytorch
                data_loader.sampler.set_epoch(runner.epoch)
            elif hasattr(data_loader.batch_sampler.sampler, 'set_epoch'):
                # batch sampler in pytorch wraps the sampler as its attributes.
                data_loader.batch_sampler.sampler.set_epoch(runner.epoch)


class DualOmniSourceRunner(DualEpochBasedRunner):
    """OmniSource Epoch-based Runner.

    This runner train models epoch by epoch, the epoch length is defined by the
    dataloader[0], which is the main dataloader.
    """

    def run_iter(self, rgb_data_batch, ske_data_batch, train_mode, source, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, rgb_data_batch, ske_data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step((rgb_data_batch, ske_data_batch), self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step((rgb_data_batch, ske_data_batch), self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        # Since we have multiple sources, we add a suffix to log_var names,
        # so that we can differentiate them.
        if 'log_vars' in outputs:
            log_vars = outputs['log_vars']
            log_vars = {k + source: v for k, v in log_vars.items()}
            self.log_buffer.update(log_vars, outputs['num_samples'])

        self.outputs = outputs

    def train(self, data_loaders, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loaders = data_loaders
        self.main_loader = self.data_loaders[0]
        # Add aliasing
        self.data_loader = self.main_loader
        self.aux_loaders = self.data_loaders[1:]
        self.aux_iters = [cycle(loader) for loader in self.aux_loaders]

        auxiliary_iter_times = [1] * len(self.aux_loaders)
        use_aux_per_niter = 1
        if 'train_ratio' in kwargs:
            train_ratio = kwargs.pop('train_ratio')
            use_aux_per_niter = train_ratio[0]
            auxiliary_iter_times = train_ratio[1:]

        self._max_iters = self._max_epochs * len(self.main_loader)

        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, (rgb_data_batch, ske_data_batch) in enumerate(self.main_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(rgb_data_batch, ske_data_batch, train_mode=True, source='')
            self.call_hook('after_train_iter')

            if self._iter % use_aux_per_niter != 0:
                self._iter += 1
                continue

            for idx, n_times in enumerate(auxiliary_iter_times):
                for _ in range(n_times):
                    data_batch = next(self.aux_iters[idx])
                    self.call_hook('before_train_iter')
                    self.run_iter(
                        data_batch, train_mode=True, source=f'/aux{idx}')
                    self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    # Now that we use validate hook, not implement this func to save efforts.
    def val(self, data_loader, **kwargs):
        raise NotImplementedError

class DualAnnealingRunner(DualEpochBasedRunner):
    def run_iter(self, rgb_data_batch, ske_data_batch, train_mode, **kwargs):
        
        if 'annealing' in kwargs:
            kwargs.update(epoch=self.epoch)
            kwargs.update(total_epoch=self.max_epochs)
            kwargs.update(iter=self._iter)
        
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, rgb_data_batch, ske_data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(rgb_data_batch, ske_data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(rgb_data_batch, ske_data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs