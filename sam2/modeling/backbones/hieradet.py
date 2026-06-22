# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP




def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class MaskedAdaptiveAvgPool2d(nn.Module):
    """带掩码的自适应平均池化层，数值稳定版本"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # 更安全的极小值
        
    def forward(self, x, mask):
        # 确保掩码非负
        mask = mask.clamp(0, 1)
        
        # 对齐特征图与掩码尺寸
        if mask.size(2) != x.size(2) or mask.size(3) != x.size(3):
            mask = F.interpolate(
                mask, 
                size=x.size()[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 计算有效掩码区域（数值稳定的方法）
        pixel_count = mask.sum(dim=(2, 3))[0]
        if pixel_count == 0:
            print("prompt is None")
            pixel_count = mask.size(2) * mask.size(3)
        
        weighted_sum = torch.sum(x * mask, dim=(2, 3), keepdim=True)
        result = weighted_sum / pixel_count
        
        return result


class Fg_Enhance(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # 可学习参数
        # self.alpha = nn.Parameter(torch.tensor(1.0))  
        # self.beta = nn.Parameter(torch.tensor(1.0))   # 背景抑制强度
        # self.theta = nn.Parameter(torch.tensor(1.0))   # 背景抑制强度
        
        self.linear = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False), 
            # nn.GroupNorm(1, in_channels // 2),
            # nn.Conv2d(in_channels // 2, 64, kernel_size=1, bias=False), 
            # nn.GroupNorm(1, 64),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    
    def forward(self, x, prompt_mask):
        # fg enhance
        if prompt_mask.shape[2:] != x.shape[2:]:
            prompt_mask = F.interpolate(
                prompt_mask, size=x.shape[2:], mode='bilinear', align_corners=False
            )
        
        out = self.linear(x)

        stage_prompt = torch.sigmoid(out)

        enhanced_feat = x + (x * prompt_mask)

        return enhanced_feat, stage_prompt
    
class Bg_Enhance(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 特征处理器模块（用于提取token）
        self.feature_processor = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels*2, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseSeparableConv(in_channels*2, in_channels*4, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseSeparableConv(in_channels*4, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        self.masked_pool = MaskedAdaptiveAvgPool2d()
    
    def forward(self, x, prompt_mask):
        # bg enhance
        bg_mask = 1 - prompt_mask
        bg_masked_x = x * bg_mask  # [B, C, H, W]
        # bg_token = self.feature_processor(bg_masked_x) # [B, C, H, W]
        bg_token = self.masked_pool(bg_masked_x, bg_mask) # [B, C, 1, 1]
        bg_expanded = bg_token.expand(-1, -1, x.size(2), x.size(3)) # [B, C, H, W]
        background_suppression = x - bg_expanded # [B, C, H, W]
        
        return x + background_suppression

def get_prompt_loss(pred, mask):
    # inter          = (pred*mask).sum(dim=(2,3))
    # union          = (pred+mask).sum(dim=(2,3))
    # loss_dice      = 1-(2*inter/(union+1)).mean()
    # return loss_dice
    return -(mask * torch.log(pred + 1e-6) + (1 - mask) * torch.log(1 - pred + 1e-6)).mean()

def to_uint8_binary(mask):
    """将浮点掩码转换为二值 uint8 图像"""
    # 转换为0-255范围
    mask_np = mask.detach().cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    return mask_np

def get_coords_list(np_prompt):
    coords_list = []
    for k in range(np_prompt.shape[0]):
        contours = cv2.findContours(np_prompt[k], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        coords = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = int(x), int(y), int(w), int(h)
            coord = [max(x,0), max(y, 0), min(x+w, np_prompt.shape[2]), min(y+h, np_prompt.shape[1])]
            coords.append(coord)
        coords_list.append(coords)
    return coords_list

def M2B(stage_prompt, coords_list):
    pred = torch.cat([stage_prompt],dim=0)
    for i in range(pred.shape[0]):
        coords = coords_list[i]
        for coord in coords:
            t_x, t_y, b_x, b_y = coord
            predW, predH = pred[i, 0, t_y:b_y, t_x:b_x].max(dim=0, keepdim=True)[0], pred[i, 0, t_y:b_y, t_x:b_x].max(dim=1, keepdim=True)[0]
            pred[i, 0, t_y:b_y, t_x:b_x] = torch.minimum(predW, predH)
    return pred


class Hiera(nn.Module):
    """
    修改后的Hiera模型，集成阶段提示编码和差分增强
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        window_spec: Tuple[int, ...] = (8, 4, 14, 7),
        global_att_blocks: Tuple[int, ...] = (12, 16, 20),
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        # 初始化原始Hiera结构
        self.window_spec = window_spec
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.global_att_blocks = global_att_blocks

        # 位置嵌入
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        cur_stage = 1
        self.blocks = nn.ModuleList()

        # 初始化Hiera块
        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        """生成位置嵌入"""
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    # def forward(
    #     self, 
    #     x: torch.Tensor, 
    #     prompt_mask: torch.Tensor = None,
    #     mode = None
    # ) -> List[torch.Tensor]:
    #     # 初始化默认提示掩码 (如果没有提供)
    #     if prompt_mask is None:
    #         print("prompt is None: ----------------------")
    #         prompt_mask = torch.zeros(
    #             (x.shape[0], 1, x.shape[2], x.shape[3]), 
    #             dtype=x.dtype, device=x.device
    #         )
    #     elif prompt_mask.dim() == 3:
    #         prompt_mask = prompt_mask.unsqueeze(1).float()
              
    #     np_prompt = to_uint8_binary(prompt_mask[:,0])
    #     coords_list = get_coords_list(np_prompt)

    #     # 特征提取
    #     x = self.patch_embed(x)  # [B, H, W, C]
    #     x = x + self._get_pos_embed(x.shape[1:3])  # 添加位置编码
        
    #     outputs = []
    #     stage_idx = 0  # 阶段索引
    #     GT_prompt = prompt_mask
    #     prompt_loss = 0
    #     for i, blk in enumerate(self.blocks):
    #         x = blk(x)
    #         # 在每个阶段结束时处理
    #         if i in self.stage_ends:

    #             x_feat = x.permute(0, 3, 1, 2)
    #             enhanced_feat, stage_prompt = self.fg_enhance_modules[stage_idx](x_feat, prompt_mask)
    #             if stage_idx in [2,3]: 
    #                 enhanced_feat = self.bg_enhance_modules[stage_idx](enhanced_feat, stage_prompt)
    #             stage_prompt = F.interpolate(stage_prompt, size=GT_prompt.shape[2:], mode='bilinear', align_corners=False)
    #             M2B_stage_prompt = M2B(stage_prompt, coords_list)
    #             prompt_loss += get_prompt_loss(M2B_stage_prompt, GT_prompt)
    #             prompt_mask = stage_prompt
    #             save_iamge(prompt_mask[0,0].detach().cpu(), 'prompt'+str(i), 0)
    #             x = enhanced_feat.permute(0, 2, 3, 1)
                
    #             # 保存增强后的特征
    #             outputs.append(enhanced_feat)
    #             stage_idx += 1

    #     return outputs, prompt_loss / (stage_idx+1)
    # return blk(x)
    