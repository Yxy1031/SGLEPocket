from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

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



class MESA(nn.Module):
    def __init__(self,inputnumchannel, kernel_size=7, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.conv = nn.Conv3d(3, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.mamba_layer = MambaLayer(
            input_dim=inputnumchannel,      
            output_dim=1,     
            d_state=d_state,  
            d_conv=d_conv,   
            expand=expand  
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, D, H, W]
        avg_result = torch.mean(x, dim=1, keepdim=True)    # [B, 1, D, H, W]
        xm=self.mamba_layer(x)
        result = torch.cat([max_result, avg_result,xm], 1)     # [B, 2, D, H, W]
        output = self.conv(result)                          # [B, 1, D, H, W]拼接之后维度转换为1
        output = self.sigmoid(output)
        return output
    
class SEMG(nn.Module):
    def __init__(self, F_g, F_l):
        super(SEMG, self).__init__()

        self.W_g = MESA(inputnumchannel=F_g)

        self.W_x = MESA(inputnumchannel=F_l)

        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.sigmoid(g1 + x1)
        return x *psi
    