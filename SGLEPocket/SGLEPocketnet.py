from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode

from mamba_ssm import Mamba
# 添加注意力模块导入
from SGLEPocket.agmodule  import SEMG


def get_dwconv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels, 
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, 
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)

class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_mamba_layer(
    spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1
):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims==2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims==3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer


class ResMambaBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )
        self.conv2 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x


class ResUpBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.skip_scale= nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)
        return x

class LFE(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        act: tuple | str = ("RELU", {"inplace": True}),
    ):
        super().__init__()
        
        if isinstance(norm, tuple) and norm[0].lower() == "group":
            num_groups = norm[1].get("num_groups", 1)
            mid_channels = in_channels // 2
            if mid_channels % num_groups != 0:
                mid_channels = ((mid_channels // num_groups) + 1) * num_groups
        else:
            mid_channels = in_channels // 2
            
        self.LFE_detector = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, mid_channels, kernel_size=3),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=mid_channels),
            get_act_layer(act),
            get_conv_layer(spatial_dims, mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.feature_enhancer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels + 1, in_channels, kernel_size=3),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels),
            get_act_layer(act)
        )
        
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        LFE_map = self.LFE_detector(x)
        
        x_with_LFE = torch.cat([x, LFE_map], dim=1)
        enhancement = self.feature_enhancer(x_with_LFE)
        

        return x + self.residual_scale * enhancement


class SGLEPocket(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 1, 1),
        blocks_bottleneck: int = 2,
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
        use_LFE: bool = True,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_bottleneck = blocks_bottleneck
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.use_LFE = use_LFE
        
        self.convInit = get_dwconv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        
        if self.use_LFE:
            self.LFE_enhancers = self._make_LFE_enhancers()
        
        self.bottleneck = self._make_bottleneck_layer()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

        self.adjust = nn.ConvTranspose3d(
            in_channels=48,
            out_channels=48,
            kernel_size=3,
            stride=2,
            padding=0,
            output_padding=0
        )
    
        self.channel_reduce = get_conv_layer(
            spatial_dims=3,
            in_channels=48,
            out_channels=24,
            kernel_size=1
        )

        self.attention_gates = nn.ModuleList()
        n_up = len(blocks_up)

        for i in range(n_up):
            if i != 2: 
                sample_in_channels = init_filters * 2 ** (n_up - i)
                self.attention_gates.append(
                    SEMG(
                        F_g=sample_in_channels // 2,  
                        F_l=sample_in_channels // 2,  
                    )
                )
        
        self.attention_special = SEMG(
            F_g=24,  
            F_l=24, 
        )

    def _make_LFE_enhancers(self):
        enhancers = nn.ModuleList()
        filters = self.init_filters
        
        for i in range(len(self.blocks_down)):
            channels = filters * 2**i
            enhancer = LFE(
                self.spatial_dims,
                channels,
                norm=self.norm,
                act=self.act
            )
            enhancers.append(enhancer)
            
        return enhancers

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            downsample_mamba = (
                get_mamba_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba, 
                *[ResMambaBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers
    
    def _make_bottleneck_layer(self):
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        
        i = len(blocks_down)
        layer_in_channels = filters * 2**i
        
        downsample_mamba = get_mamba_layer(
            spatial_dims, 
            layer_in_channels // 2, 
            layer_in_channels, 
            stride=2
        )
        
        bottleneck_layer = nn.Sequential(
            downsample_mamba,
            *[ResMambaBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) 
            for _ in range(self.blocks_bottleneck)]
        )
        
        return bottleneck_layer

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResUpBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_dwconv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
            
        down_x = []
        
        for i, down in enumerate(self.down_layers):

            x = down(x)
            if self.use_LFE:
                x = self.LFE_enhancers[i](x)
            
            down_x.append(x)
        
        return x, down_x
    
    def bottleneck_forward(self, x: torch.Tensor) -> torch.Tensor:
        """瓶颈层处理"""
        x = self.bottleneck(x)
        return x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        """解码器"""
        attention_idx = 0
        
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            if i == 2:  
                x = self.adjust(x)
                x = self.channel_reduce(x)

                skip_feature = down_x[i + 1]
                att_skip_feature = self.attention_special(x, skip_feature)  
                x = x + att_skip_feature
                x = upl(x)
            else:
                up_feature = up(x)
                skip_feature = down_x[i + 1]
                att_skip_feature = self.attention_gates[attention_idx](up_feature, skip_feature)#g,x
                attention_idx += 1
                x = up_feature + att_skip_feature
                x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器
        x, down_x = self.encode(x)
        
        # 瓶颈层
        x = self.bottleneck_forward(x)
        down_x.append(x)
    
        down_x.reverse()
        
        # 解码器
        x = self.decode(x, down_x)
        x = torch.sigmoid(x)
        return x
