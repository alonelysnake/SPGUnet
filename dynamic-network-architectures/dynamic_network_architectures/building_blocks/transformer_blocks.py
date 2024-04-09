from typing import Tuple, List, Union, Type, Sequence, Any, Iterable
import math
import os

import numpy as np
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

def ensure_tuple_rep(tup: Any, dim: int) -> Tuple[Any, ...]:
    """
    参见monai
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    def issequenceiterable(obj: Any) -> bool:
        """
        Determine if the object is an iterable sequence and is not a string.
        """
        try:
            if hasattr(obj, "ndim") and obj.ndim == 0:
                return False  # a 0-d tensor is not iterable
        except Exception:
            return False
        return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    参见monai的trunc_normal_
    """
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4, pos_embed="conv")

    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int]=(96,288,80),#TODO 此处如何修改成不是预设?
        patch_size: Union[Sequence[int], int]=(16,16,16),
        hidden_size: int=768,
        num_heads: int=12,
        pos_embed: str = 'conv',#目前代码只支持perceptron的
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size. 保证img_size被patch_size整除
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.暂不使用
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.pos_embed = 'perceptron'

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = int(in_channels * np.prod(patch_size))

        self.patch_embeddings: nn.Module
        if self.pos_embed == "conv":
            print('self.pos_embed == "conv" 代码未完成, ',os.path.abspath(os.path.dirname(__file__)))
            exit(0)
            # self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
            #     in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            # )
        elif self.pos_embed == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embeddings(x)
        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class MLPBlock(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        # act: Union[Tuple, str] = "GELU",
        # dropout_mode="vit",
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: faction of the input units to drop.
            act: activation type and arguments. Defaults to GELU.
            dropout_mode: dropout mode, can be "vit" or "swin".
                "vit" mode uses two dropout instances as implemented in
                https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
                "swin" corresponds to one instance as implemented in
                https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)# 或者和drop1共用
        # dropout_opt = look_up_option(dropout_mode, SUPPORTED_DROPOUT_MODE)
        # if dropout_opt == "vit":
        #     self.drop2 = nn.Dropout(dropout_rate)
        # elif dropout_opt == "swin":
        #     self.drop2 = self.drop1
        # else:
        #     raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self,
                 hidden_size: int, 
                 num_heads: int, 
                 dropout_rate: float = 0.0) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

    def forward(self, x):
        output = self.input_rearrange(self.qkv(x)) # b patch_num hidden_size -> b h hidden_size*3 -> 3 b num_heads patch_num dim_per_head
        q, k, v = output[0], output[1], output[2] # b num_heads patch_num dim_per_head
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1) # b num_heads patch_num patch_num
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v) # b num_heads patch_num patch_num -> b num_heads patch_num dim_per_head
        x = self.out_rearrange(x) # b num_heads patch_num dim_per_head -> b patch_num hidden_size
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x

class MySABlock(SABlock):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0) -> None:
        super().__init__(hidden_size, num_heads, dropout_rate)
        self.input_rearrange = Rearrange("b qkv h (l d) -> qkv b l h d", l=num_heads)
    
    def forward(self, x):
        # x.shape = [b, qkv=3, patch_num, hidden_size]
        # print('MySABlock shape should be [b, qkv=3, patch_num, hidden_size], got', x.shape)
        output = self.input_rearrange(x)
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1) # b num_heads patch_num patch_num
        att_mat = self.drop_weights(att_mat)
        output = torch.einsum("bhxy,bhyd->bhxd", att_mat, v) # b num_heads patch_num patch_num -> b num_heads patch_num dim_per_head
        output = self.out_rearrange(output) # b num_heads patch_num dim_per_head -> b patch_num hidden_size
        output = self.out_proj(output)
        output = self.drop_output(output)
        return output


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        # spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        # norm_name: Union[Tuple, str],
        # act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            stride=stride,
        )
        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            stride=1,
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels)
        # self.conv1 = get_conv_layer(
        #     spatial_dims,
        #     in_channels,
        #     out_channels,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     dropout=dropout,
        #     conv_only=True,
        # )
        # self.conv2 = get_conv_layer(
        #     spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        # )
        # self.lrelu = get_act_layer(name=act_name)
        # self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        # self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels 
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride
            )
            self.norm3 = nn.InstanceNorm3d(num_features=out_channels)
            # self.conv3 = get_conv_layer(
            #     spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True
            # )
            # self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out

class TransformerPrUpBlock(nn.Module):
    """
    projection upsampling module
    从transformer的输出(output_channel=hidden_size)到feature size
    """
    def __init__(
        self,
        in_channels: int,# hidden_size
        out_channels: int,# feature size
        num_layer: int=3,#除了第一次upsample外进行upsample的层数,考虑设置为3?(patch_size=16, log(16)-1)
        kernel_size: Union[Sequence[int], int]=3,#固定为3
        stride: Union[Sequence[int], int]=1,# 固定为1
        upsample_kernel_size: Union[Sequence[int], int]=2,# 固定为2
        # spatial_dims: int,# 固定为3
        # norm_name: Union[Tuple, str], # 固定为torch.nn.modules.instancenorm.InstanceNorm3d
        conv_block: bool = False, # 改设为True
        res_block: bool = False, # 改设为True
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_layer: number of upsampling blocks.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        upsample_stride = upsample_kernel_size
        # TODO 同时上采样并改变通道数，是否不利于学习?
        self.transp_conv_init = torch.nn.ConvTranspose3d(
            in_channels=in_channels,#768,transformer的hidden_size
            out_channels=out_channels,#16,nnunet stages[1]的out_channel//2
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            bias=False
        )
        # self.transp_conv_init = get_conv_layer(
        #     spatial_dims,
        #     in_channels,
        #     out_channels,
        #     kernel_size=upsample_kernel_size,
        #     stride=upsample_stride,
        #     conv_only=True,
        #     is_transposed=True,
        # )# torch.nn.modules.conv.ConvTranspose3d即可
        # 下面的block个数用来满足最后输出的vol.shape和没有transformer的一样(对应于transformer的patch_size)
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            torch.nn.ConvTranspose3d(
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                bias=False
                            ),
                            UnetResBlock(
                                # spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                # norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
            else:
                print("代码未完成1, TransformerPrUpBlock.init", os.path.abspath(os.path.dirname(__file__)))
                exit(0)
                # self.blocks = nn.ModuleList(
                #     [
                #         nn.Sequential(
                #             get_conv_layer(
                #                 spatial_dims,
                #                 out_channels,
                #                 out_channels,
                #                 kernel_size=upsample_kernel_size,
                #                 stride=upsample_stride,
                #                 conv_only=True,
                #                 is_transposed=True,
                #             ),
                #             UnetBasicBlock(
                #                 spatial_dims=spatial_dims,
                #                 in_channels=out_channels,
                #                 out_channels=out_channels,
                #                 kernel_size=kernel_size,
                #                 stride=stride,
                #                 norm_name=norm_name,
                #             ),
                #         )
                #         for i in range(num_layer)
                #     ]
                # )
        else:
            print("代码未完成2, TransformerPrUpBlock.init", os.path.abspath(os.path.dirname(__file__)))
            exit(0)
            # self.blocks = nn.ModuleList(
            #     [
            #         get_conv_layer(
            #             spatial_dims,
            #             out_channels,
            #             out_channels,
            #             kernel_size=upsample_kernel_size,
            #             stride=upsample_stride,
            #             conv_only=True,
            #             is_transposed=True,
            #         )
            #         for i in range(num_layer)
            #     ]
            # )

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 mlp_dim,
                 dropout_rate,
                 num_heads):
        
        super().__init__()

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # x.shape = [b, 1, patch_num, hidden_size]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MyTransformerLayer(TransformerLayer):
    def __init__(self, hidden_size, mlp_dim, dropout_rate, num_heads):
        super().__init__(hidden_size, mlp_dim, dropout_rate, num_heads)

        self.attn = MySABlock(hidden_size, num_heads, dropout_rate)
        # self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # x.shape = [b, qkv=3, patch_num, hidden_size]
        # print('MyTransformerLayer shape should be [b, qkv=3, patch_num, hidden_size], got', x.shape)
        # x[:,0]=q, x[:,1]=k, x[:,2]= v
        new_xv = x[:,2,...] + self.attn(self.norm1(x)) # 只更新v
        new_xv = new_xv + self.mlp(self.norm2(new_xv))
        output = torch.stack([x[:,0], x[:,1], new_xv],dim=1)
        return output
        # 等价于下面的(不能原地修改)
        # x[:,2,...] = x[:,2,...] + self.attn(self.norm1(x))
        # x[:,2,...] = x[:,2,...] + self.mlp(self.norm2(x[:,2,...]))
        # return x

class MyTransformerBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 num_layers: int = 12,#TODO 不需要那么多层，改成3
                 img_size: Union[Sequence[int], int]=(96,288,80),
                 patch_size: Union[Sequence[int], int] =(16, 16, 16),
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 num_heads: int = 1,# TODO vit预设为12
                 pos_embed: str = "conv",
                 classification: bool = False,
                 dropout_rate: float = 0.0,
                 spatial_dims: int = 3,
                #  num_classes: int = 2
                 ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size, 
                mlp_dim=mlp_dim,
                num_heads=num_heads, 
                dropout_rate=dropout_rate
            ) for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        # if self.classification:
        #     self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        #     self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
        
        # 从transformer的输出变成unet的stages[0]的输出
        self.encoder = TransformerPrUpBlock(
            in_channels=hidden_size,
            out_channels=out_channels,
            num_layer=3,
            conv_block=True,
            res_block=True
        )


    def forward(self, x):
        x = self.patch_embedding(x) # ? -> b patch_num hidden_size
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        # hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
        #     hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        x = self.encoder(self.proj_feat(x, self.hidden_size, self.feat_size))
        return x#, hidden_states_out# 只返回x即可
    
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def compute_conv_feature_map_size(self, input_size):
        # 计算参数个数
        output = self.patch_embedding.compute_conv_feature_map_size(input_size)
        # embedding_size = np.prod(self.feat_size)
        encode_size = input_size.copy()
        encode_size[-3:]=self.feat_size
        encode_size[-4]=self.hidden_size
        output += self.encoder.compute_conv_feature_map_size(encode_size)
        return 0
    

class MyTransformerBlock1(MyTransformerBlock):
    """
    固定qkv, q v为CT, k为supervoxel
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 num_layers: int = 12, 
                 img_size: Union[Sequence[int], int]=(96,288,80),
                 patch_size: Union[Sequence[int], int] =(16, 16, 16),
                 hidden_size: int = 768, 
                 mlp_dim: int = 3072, 
                 num_heads: int = 1, 
                 pos_embed: str = "conv", 
                 classification: bool = False, 
                 dropout_rate: float = 0, 
                 spatial_dims: int = 3
                 ) -> None:
        super().__init__(in_channels, out_channels, num_layers, img_size, patch_size, hidden_size, mlp_dim, num_heads, pos_embed, classification, dropout_rate, spatial_dims)
        
        self.patch_embeddings=nn.ModuleList([
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                pos_embed=pos_embed,
                dropout_rate=dropout_rate,
                spatial_dims=spatial_dims,
            ) for i in range(3)
        ]) # 每个通道分别embedding TODO 目前只支持ct和supervoxel双通道的

        self.blocks = nn.ModuleList([
            MyTransformerLayer(
                hidden_size=hidden_size, 
                mlp_dim=mlp_dim,
                num_heads=num_heads, 
                dropout_rate=dropout_rate
            ) for i in range(num_layers)
        ])
    
    def forward(self, x):
        # TODO 目前只针对channel=2
        # x = torch.stack([x[:,0,...], x[:,1,...], x[:,0,...]], dim=1) # x.shape = [b, qkv=3, patch_num, hidden_size] TODO 此处q是否应该为新的拷贝?
        # x = self.patch_embedding(x) # ? -> b patch_num hidden_size
        # TODO 注意修改qkv对应的channel
        embeddings = []
        embeddings.append(self.patch_embeddings[0](x[:,1:2])) # q
        embeddings.append(self.patch_embeddings[1](x[:,0:1])) # k
        embeddings.append(self.patch_embeddings[2](x[:,0:1])) # v
        x = torch.stack(embeddings, dim=1) # x.shape = [b, qkv, patch_num, hidden_size]
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        # hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
        #     hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        x = x[:,2] # 只保留融合后的v
        x = self.encoder(self.proj_feat(x, self.hidden_size, self.feat_size))
        return x#, hidden_states_out# 只返回x即可



if __name__=="__main__":
    x=np.random.random([2,3,96,288,80])
    x = torch.Tensor(x)
    x = x.to('cuda:0')
    model = MyTransformerBlock1(2,32)
    model.to('cuda:0')
    model.train()
    loss = model(x).sum()
    loss.backward()
    # for n,p in model.named_parameters():
    #     print(n,p.device)
    