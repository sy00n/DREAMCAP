from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class DREAMRecognizer3D(BaseRecognizer):
    """3D recognizer model framework."""
    def forward(self, rgbs, skes, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(rgbs, skes, **kwargs)
        if kwargs.get('get_feat', False):
            del kwargs['get_feat']
            return self.get_feat(rgbs, skes, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(rgbs, skes, label, **kwargs)

        return self.forward_test(rgbs, skes, **kwargs)

    def extract_feat(self, rgbs, skes):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        x = self.backbone(rgbs, skes)
        return x

    def forward_train(self, rgbs, skes, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        rgbs = rgbs.reshape((-1, ) + rgbs.shape[2:])
        skes = skes.reshape((-1, ) + skes.shape[2:])
        losses = dict()

        x = self.extract_feat(rgbs, skes)

        if hasattr(self, 'debias_head'):
            loss_debias = self.debias_head(x, target=labels.squeeze(), **kwargs)
            losses.update(loss_debias)

        if hasattr(self, 'neck'):
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, rgbs, skes):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        num_segs = rgbs.shape[1]
        rgbs = rgbs.reshape((-1, ) + rgbs.shape[2:])
        skes = skes.reshape((-1, ) + skes.shape[2:])

        x = self.extract_feat(rgbs, skes)
        if hasattr(self, 'neck'):
            x, _ = self.neck(x)

        cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score, num_segs)

        return cls_score

    def forward_test(self, rgbs, skes):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(rgbs, skes).cpu().numpy()

    def forward_dummy(self, rgbs, skes):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        rgbs = rgbs.reshape((-1, ) + rgbs.shape[2:])
        skes = skes.reshape((-1, ) + skes.shape[2:])

        x = self.extract_feat(rgbs, skes)
        outs = (self.cls_head(x), )
        return outs

    def forward_gradcam(self, rgbs, skes):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(rgbs, skes)

    def get_feat(self, rgbs, skes, return_score=False):
        """Defines the computation performed at every call when using get_feat
        utils."""
        num_segs = rgbs.shape[1]
        rgbs = rgbs.reshape((-1, ) + rgbs.shape[2:])
        skes = skes.reshape((-1, ) + skes.shape[2:])

        x = self.extract_feat(rgbs, skes)
        if hasattr(self, 'neck'):
            x, _ = self.neck(x)  # (num_clips * num_crops, 2048, 1, 8, 8)
        
        if return_score:
            cls_score = self.cls_head(x)
            cls_score = self.average_clip(cls_score, num_segs)
            return x, cls_score
        return x
    
    def train_step(self, rgb_data_batch, ske_data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        rgb_imgs = rgb_data_batch['imgs']
        ske_imgs = ske_data_batch['imgs']
        rgb_label = rgb_data_batch['label']
        ske_label = ske_data_batch['label']
        
        assert rgb_label.shape[0] == sum(rgb_label==ske_label)[0]

        aux_info = {}
        for item in self.aux_info:
            assert item in rgb_data_batch
            aux_info[item] = rgb_data_batch[item]
        aux_info.update(kwargs)

        losses = self(rgb_imgs, ske_imgs, rgb_label, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(rgb_data_batch.values()))))

        return outputs
    
    def val_step(self, rgb_data_batch, ske_data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        rgb_imgs = rgb_data_batch['imgs']
        ske_imgs = ske_data_batch['imgs']
        rgb_label = rgb_data_batch['label']
        ske_label = ske_data_batch['label']

        assert rgb_label.shape[0] == sum(rgb_label==ske_label)[0]
        
        aux_info = {}
        for item in self.aux_info:
            aux_info[item] = rgb_data_batch[item]
 
        losses = self(rgb_imgs, ske_imgs, rgb_label, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(rgb_data_batch.values()))))

        return outputs