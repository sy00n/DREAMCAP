import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import print_log

from ...utils import get_root_logger
from ..registry import BACKBONES
from .resnet3d import ResNet3d


class ResNet3dPathway(ResNet3d):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        *args (arguments): Arguments same as :class:``ResNet3d``.
        lateral (bool): Determines whether to enable the lateral connection
            from another pathway. Default: False.
        fusion_kernel (int): The kernel size of lateral fusion.
            Default: 5.
        **kwargs (keyword arguments): Keywords arguments for ResNet3d.
    """

    def __init__(self,
                 *args,
                 lateral=False,
                 fusion_kernel=5,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 inflate=(1, 1, 1, 1),
                 with_cp=False,
                 lateral_channels=None,
                 lateral_expansion=1,
                 **kwargs):
        self.lateral = lateral
        self.fusion_kernel = fusion_kernel
        super().__init__(*args, **kwargs)
        if self.lateral:
            self.lateral_channels = lateral_channels
        else:
            self.lateral_channels = 0
        self.inplanes = self.base_channels

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                non_local=self.non_local_stages[i],
                non_local_cfg=self.non_local_cfg,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            if i != 0:
                self.lateral_channels *= 2
            else:
                self.lateral_channels *= lateral_expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def make_res_layer(self,
                       block,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       style='pytorch',
                       inflate=1,
                       inflate_style='3x1x1',
                       non_local=0,
                       non_local_cfg=dict(),
                       conv_cfg=None,
                       norm_cfg=None,
                       act_cfg=None,
                       with_cp=False):
        """Build residual layer for Slowfast.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input
                feature in each block.
            planes (int): Number of channels for the output
                feature in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides
                in residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``,
                the stride-two layer is the 3x3 conv layer,
                otherwise the stride-two layer is the first 1x1 conv layer.
                Default: ``pytorch``.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): ``3x1x1`` or ``1x1x1``. which determines
                the kernel sizes and padding strides for conv1 and
                conv2 in each block. Default: ``3x1x1``.
            non_local (int | Sequence[int]): Determine whether to apply
                non-local module in the corresponding block of each stages.
                Default: 0.
            non_local_cfg (dict): Config for non-local module.
                Default: ``dict()``.
            conv_cfg (dict | None): Config for conv layers. Default: None.
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save
                some memory while slowing down the training speed.
                Default: False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        non_local = non_local if not isinstance(
            non_local, int) else (non_local, ) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks

        try:
            self.lateral_channels
        except:
            return None

        if (spatial_stride != 1
                or (inplanes + self.lateral_channels) != planes * block.expansion):
            downsample = ConvModule(
                inplanes + self.lateral_channels,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
        else:
            downsample = None

        layers = []
        layers.append(
            block(
                inplanes + self.lateral_channels,
                planes,
                spatial_stride,
                temporal_stride,
                dilation,
                downsample,
                style=style,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp))

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    1,
                    1,
                    dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp))

        return nn.Sequential(*layers)


pathway_cfg = {
    'resnet3d': ResNet3dPathway,
    # TODO: BNInceptionPathway
}


def build_pathway(cfg, *args, **kwargs):
    """Build pathway.

    Args:
        cfg (None or dict): cfg should contain:
            - type (str): identify conv layer type.

    Returns:
        nn.Module: Created pathway.
    """
    if not (isinstance(cfg, dict) and 'type' in cfg):
        raise TypeError('cfg must be a dict containing the key "type"')
    cfg_ = cfg.copy()

    pathway_type = cfg_.pop('type')
    if pathway_type not in pathway_cfg:
        raise KeyError(f'Unrecognized pathway type {pathway_type}')

    pathway_cls = pathway_cfg[pathway_type]
    pathway = pathway_cls(*args, **kwargs, **cfg_)

    return pathway


@BACKBONES.register_module()
class DREAM(nn.Module):
    """Slowfast backbone.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        pretrained (str): The file path to a pretrained model.
        rgb_pathway (dict): Configuration of slow branch, should contain
            necessary arguments for building the specific type of pathway
            and:
            type (str): type of backbone the pathway bases on.
            lateral (bool): determine whether to build lateral connection
            for the pathway.Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=True, depth=50, pretrained=None,
                conv1_kernel=(1, 7, 7), dilations=(1, 1, 1, 1),
                conv1_stride_t=1, pool1_stride_t=1, inflate=(0, 0, 1, 1))

        ske_pathway (dict): Configuration of fast branch, similar to
            `rgb_pathway`. Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=False, depth=50, pretrained=None, base_channels=8,
                conv1_kernel=(5, 7, 7), conv1_stride_t=1, pool1_stride_t=1)
    """

    def __init__(self,
                 pretrained,
                 rgb_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=True,
                     conv1_kernel=(1, 7, 7),
                     dilations=(1, 1, 1, 1),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     inflate=(0, 0, 1, 1)),
                 ske_pathway=dict(
                     type='resnet3d',
                     depth=34,
                     pretrained=None,
                     lateral=False,
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1)):
        super().__init__()
        self.pretrained = pretrained

        if rgb_pathway['lateral']:
            rgb_pathway['lateral_channels'] = ske_pathway['base_channels']
            if ske_pathway['depth'] > 34:
                rgb_pathway['lateral_expansion'] = 4
        if ske_pathway['lateral']:
            ske_pathway['lateral_channels'] = rgb_pathway['base_channels']
            if rgb_pathway['depth'] > 34:
                ske_pathway['lateral_expansion'] = 4
        self.rgb_path = build_pathway(rgb_pathway)
        self.ske_path = build_pathway(ske_pathway)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            # Directly load 3D model.
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init two branch seperately.
            self.ske_path.init_weights()
            self.rgb_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, rgb, ske):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted
                by the backbone.
        """
        
        x_rgb = self.rgb_path.conv1(rgb)
        x_rgb = self.rgb_path.maxpool(x_rgb)

        x_ske = self.ske_path.conv1(ske)
        x_ske = self.ske_path.maxpool(x_ske)

        if self.rgb_path.lateral:
            x_rgb = torch.cat((x_rgb, x_ske), dim=1)

        for i, layer_name in enumerate(self.rgb_path.res_layers):
            res_layer = getattr(self.rgb_path, layer_name)
            x_rgb = res_layer(x_rgb)
            res_layer_ske = getattr(self.ske_path, layer_name)
            x_ske = res_layer_ske(x_ske)
            if (i != len(self.rgb_path.res_layers) - 1
                    and self.rgb_path.lateral):
                x_rgb = torch.cat((x_rgb, x_ske), dim=1)

        out = (x_rgb, x_ske)

        return out