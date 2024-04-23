from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import numpy as np

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder, SuperVoxelDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from dynamic_network_architectures.architectures.unet import PlainConvUNet


class SPGUNet(PlainConvUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__(input_channels - 1,  # one stage is supervoxel
                         n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder,
                         conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, nonlin_first)
        self.final_decoder = SuperVoxelDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first)
        self.stage = 1 # default is the final stage, use set_stage to 0 if train from raw

    def forward(self, x):
        # CT, cal, supervoxel
        x = x[:, 0:-1]
        if self.stage == 0:
            supervoxel = None
            skips = self.encoder(x)
        else:
            supervoxel = x[:, -1:]
            with torch.no_grad():
                skips = self.encoder(x)

        if supervoxel is None:
            return self.decoder(skips)
        else:
            return self.final_decoder(skips, supervoxel=supervoxel)

    def set_stage(self, stage):
        self.stage = stage


if __name__ == '__main__':
    data = torch.rand((1, 1, 64, 64, 64))

    model = SPGUNet(input_channels=1,
                    n_stages=6,
                    features_per_stage=(32, 64, 125, 256, 320, 320),
                    conv_op=nn.Conv3d,
                    kernel_sizes=3,
                    strides=(1, 2, 2, 2, 2, 2),
                    n_conv_per_stage=(2, 2, 2, 2, 2, 2),
                    num_classes=4,
                    n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
                    conv_bias=False, norm_op=nn.BatchNorm3d,
                    norm_op_kwargs=None,
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nn.ReLU,
                    deep_supervision=True)
    model(data)
    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))

    # data = torch.rand((1, 4, 512, 512))
    #
    # model = PlainConvUNet(4, 8, (32, 64, 125, 256, 512, 512, 512, 512), nn.Conv2d, 3, (1, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2), 4,
    #                             (2, 2, 2, 2, 2, 2, 2), False, nn.BatchNorm2d, None, None, None, nn.ReLU, deep_supervision=True)
    #
    # if False:
    #     import hiddenlayer as hl
    #
    #     g = hl.build_graph(model, data,
    #                        transforms=None)
    #     g.save("network_architecture.pdf")
    #     del g
    #
    # print(model.compute_conv_feature_map_size(data.shape[2:]))
