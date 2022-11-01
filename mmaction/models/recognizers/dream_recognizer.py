from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class DREAMRecognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, rgbs, skes, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        rgbs = rgbs.reshape((-1, ) + imgs.shape[2:])
        skes = skes.reshape((-1, ) + imgs.shape[2:])
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