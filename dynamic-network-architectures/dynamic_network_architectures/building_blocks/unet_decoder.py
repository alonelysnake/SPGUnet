import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, List, Tuple
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.my_plain_conv_encoder import MyPlainConvEncoder


class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder, MyPlainConvEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = [] # 0-4自下而上对应
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]#
            input_features_skip = encoder.output_channels[-(s + 1)]#skip connect的channel数（也即上采样输出的channel数）
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1] # 前一个stage的输出
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input) # 先上采样前一个stage的输出
            x = torch.cat((x, skips[-(s+2)]), 1) # 拼接同层的skip connect
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1] # 反向，变为自顶向下

        if not self.deep_supervision:
            r = seg_outputs[0] # 不多尺度监督只输出最顶层
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


class ImplicitDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        assert not self.deep_supervision, '隐式decoder不用multi-scale的loss，deep_supervision应该为false'
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        # 考虑用ios net结构的decoder
        feature_num = sum(encoder.output_channels)
        hidden_dim = 256
        self.fc_0 = nn.Conv1d(feature_num, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, num_classes, 1)
        self.actvn = nn.ReLU()

        # 以下为原nnunet的
        # transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        # we start with the bottleneck and work out way up
        # stages = []
        # transpconvs = []
        # seg_layers = []
        # for s in range(1, n_stages_encoder):
        #     input_features_below = encoder.output_channels[-s]#
        #     input_features_skip = encoder.output_channels[-(s + 1)]#skip connect的channel数（也即上采样输出的channel数）
        #     stride_for_transpconv = encoder.strides[-s]
        #     transpconvs.append(transpconv_op(
        #         input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
        #         bias=encoder.conv_bias
        #     ))
        #     # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
        #     stages.append(StackedConvBlocks(
        #         n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
        #         encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
        #         encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
        #     ))
        #
        #     # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
        #     # then a model trained with deep_supervision=True could not easily be loaded at inference time where
        #     # deep supervision is not needed. It's just a convenience thing
        #     seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
        #
        # self.stages = nn.ModuleList(stages)
        # self.transpconvs = nn.ModuleList(transpconvs)
        # self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips, points):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        # skips[0]为size最大的，3为最小的
        #TODO points.shape?要变成[batch,1,1,num_sample_points,3]
        # TODO 再加上前后左右上下的一点偏移，变成[batch,1,7,num_sample,3]，见ios net的displacement
        features = []
        for sc in skips:
            features.append(F.grid_sample(sc, points))#TODO sc.shape应为[B,C,h,w,d]
        features = torch.cat(features,dim=1) # TODO features.shape应为[B,features,1,7,num_sample]
        features = torch.reshape(features, (features.shape[0], features.shape[1] * features.shape[3], features.shape[4]))  # (B, featues_per_sample, samples_num)
        # TODO 对position的encoding

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net) # shape = [B,num_class,sample_num]

        assert not self.deep_supervision
        return net


        # lres_input = skips[-1]
        # seg_outputs = []
        # for s in range(len(self.stages)):
        #     x = self.transpconvs[s](lres_input)
        #     x = torch.cat((x, skips[-(s+2)]), 1)
        #     x = self.stages[s](x)
        #     if self.deep_supervision:
        #         seg_outputs.append(self.seg_layers[s](x))
        #     elif s == (len(self.stages) - 1):
        #         seg_outputs.append(self.seg_layers[-1](x))
        #     lres_input = x
        #
        # # invert seg outputs so that the largest segmentation prediction is returned first
        # seg_outputs = seg_outputs[::-1]
        #
        # if not self.deep_supervision:
        #     r = seg_outputs[0]
        # else:
        #     r = seg_outputs
        # return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output
    

class SuperVoxelDecoder(UNetDecoder):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder, MyPlainConvEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        super().__init__(encoder,num_classes,n_conv_per_stage,deep_supervision)

        n_stages_encoder = len(self.encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = [] # 0-4自下而上对应
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]#
            input_features_skip = encoder.output_channels[-(s + 1)]#skip connect的channel数（也即上采样输出的channel数）
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            if s != n_stages_encoder - 1:
                stages.append(StackedConvBlocks(
                    n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                    encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
                ))
            else:
                # TODO 最后一层，增加supervoxel的channel
                supervoxel_channel = 32
                stages.append(StackedConvBlocks(
                    n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip + supervoxel_channel, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                    encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
                ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

        self.supervoxel_conv = StackedConvBlocks(
            2, encoder.conv_op, 1, 32, encoder.kernel_sizes[0], 1,
            encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs, 
            encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
        )


    def forward(self, skips, supervoxel=None):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1] # 前一个stage的输出
        seg_outputs = []
        supervoxel = self.supervoxel_conv(supervoxel)
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input) # 先上采样前一个stage的输出
            if s != len(self.stages) - 1 or supervoxel==None:
                x = torch.cat((x, skips[-(s+2)]), 1) # 拼接同层的skip connect
            else:
                x = torch.cat((x, skips[-(s+2)], supervoxel), 1) # 拼接同层的skip connect 并 加上supervoxel的特征
            x = self.stages[s](x)
            # 是否多尺度监督
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1] # 反向，变为自顶向下

        if not self.deep_supervision:
            r = seg_outputs[0] # 不多尺度监督只输出最顶层
        else:
            r = seg_outputs
        return r