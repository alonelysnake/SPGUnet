from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import numpy as np

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.my_plain_conv_encoder import MyPlainConvEncoder,MyTransformerEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder, ImplicitDecoder, SuperVoxelDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from dynamic_network_architectures.architectures.unet import PlainConvUNet

class MyPlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],  # 每个stage的输出feature数
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],  # encoder,要先处理
                 num_classes: int,  # decoder
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],  # decoder,要先处理
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
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = MyPlainConvEncoder(input_channels=input_channels,
                                          n_stages=n_stages,
                                          features_per_stage=features_per_stage,
                                          conv_op=conv_op,
                                          kernel_sizes=kernel_sizes,
                                          strides=strides,
                                          n_conv_per_stage=n_conv_per_stage,
                                          conv_bias=conv_bias,
                                          norm_op=norm_op,
                                          norm_op_kwargs=norm_op_kwargs,
                                          dropout_op=dropout_op,
                                          dropout_op_kwargs=dropout_op_kwargs,
                                          nonlin=nonlin,
                                          nonlin_kwargs=nonlin_kwargs,
                                          return_skips=True,  # 是否返回每一个stage的结果
                                          nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)


class MyImplicitConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],  # 每个stage的输出feature数
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],  # encoder,要先处理
                 num_classes: int,  # decoder
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],  # decoder,要先处理
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
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels=input_channels,
                                          n_stages=n_stages,
                                          features_per_stage=features_per_stage,
                                          conv_op=conv_op,
                                          kernel_sizes=kernel_sizes,
                                          strides=strides,
                                          n_conv_per_stage=n_conv_per_stage,
                                          conv_bias=conv_bias,
                                          norm_op=norm_op,
                                          norm_op_kwargs=norm_op_kwargs,
                                          dropout_op=dropout_op,
                                          dropout_op_kwargs=dropout_op_kwargs,
                                          nonlin=nonlin,
                                          nonlin_kwargs=nonlin_kwargs,
                                          return_skips=True,  # 是否返回每一个stage的结果
                                          nonlin_first=nonlin_first)
        self.decoder = ImplicitDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        self.num_class = num_classes

    def forward(self, x, points=None):
        # 如果points=None,则对应测试阶段，要对所有点都采样然后预测
        if points is not None:
            skips = self.encoder(x)
            return self.decoder(skips, points)
        else:
            # x.shape = [b,c,h,w,d]
            skips = self.encoder(x)
            points = self._sample(x)  # points.shape = [b,1,1,sample_num, 3] TODO 可能一次不能全部放下，要考虑分批次

            output = self.decoder(skips, points)  # output.shape = [b, class, sample_num]
            
            print('max label', torch.max(torch.argmax(output,dim=1)))
            # 恢复成三维图像的结构
            points = points * torch.Tensor(list(x.shape[2:])).to(points.device)
            points = points[0].squeeze(0).squeeze(0) # shape=[sample_num, 3]
            points = torch.transpose(points,0,1).long()#shape=[3,sample_num]
            output_ = torch.zeros((x.shape[0], self.num_class, x.shape[-3],x.shape[-2],x.shape[-1]), dtype=torch.float16).to(output.device)
            output_[...,points[0],points[1],points[2]] = output

            return output_  # output.shape = [b,class,h,w,d]

    def _sample(self, vol):
        shape = vol.shape[-3:]
        grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]))
        coordinates_batch = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        coordinates_batch = coordinates_batch.view(-1, 3).to(vol.device)
        assert coordinates_batch.device == vol.device
        vol_shape = vol.shape[2:]
        coordinates_batch = coordinates_batch.float() / torch.Tensor(list(vol_shape)).to(vol.device)  # lesion区域归一化

        coordinates = torch.stack([coordinates_batch for i in range(vol.shape[0])]).unsqueeze(1).unsqueeze(1)
        return coordinates  # shape = [b, 1, 1, sample_num, 3]

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)


class MyTransformerUNet(nn.Module):
    """
    将第一层卷积变成双通道的transformer对齐,然后concat合并,后面和PlainConvUNet一样
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],  # 每个stage的输出feature数
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],  # encoder,要先处理
                 num_classes: int,  # decoder
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],  # decoder,要先处理
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
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = MyTransformerEncoder(input_channels=input_channels,
                                          n_stages=n_stages,
                                          features_per_stage=features_per_stage,
                                          conv_op=conv_op,
                                          kernel_sizes=kernel_sizes,
                                          strides=strides,
                                          n_conv_per_stage=n_conv_per_stage,
                                          conv_bias=conv_bias,
                                          norm_op=norm_op,
                                          norm_op_kwargs=norm_op_kwargs,
                                          dropout_op=dropout_op,
                                          dropout_op_kwargs=dropout_op_kwargs,
                                          nonlin=nonlin,
                                          nonlin_kwargs=nonlin_kwargs,
                                          return_skips=True,  # 是否返回每一个stage的结果
                                          nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)


class MySuperVoxelDecoderUNet(PlainConvUNet):
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
        super().__init__(input_channels-1, # 有一个channel是supervoxel
                         n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder,
                 conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, nonlin_first)
        self.decoder = SuperVoxelDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
    
    def forward(self, x):
        # channel 0是CT，最后一个channel 是supervoxel
        supervoxel = x[:,-1:]
        x = x[:,0:-1]
        with torch.no_grad():
            skips = self.encoder(x)

        return self.decoder(skips,supervoxel=supervoxel)


if __name__ == '__main__':
    data = torch.rand((1, 1, 64, 64, 64))

    model = MyPlainConvUNet(input_channels=1,
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
