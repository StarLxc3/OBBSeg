import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import os
import cv2
import numpy as np
import sys
from utils import *

sys.path.append("..") 
from sam2.build_sam import build_sam2

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    
class MaskedAdaptiveAvgPool2d(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  
        
    def forward(self, x, mask):
        mask = mask.clamp(0, 1)
        if mask.size(2) != x.size(2) or mask.size(3) != x.size(3):
            mask = F.interpolate(
                mask, 
                size=x.size()[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        pixel_count = mask.sum(dim=(2, 3))[0]
        if pixel_count == 0:
            print("prompt is None")
            pixel_count = mask.size(2) * mask.size(3)
        
        weighted_sum = torch.sum(x * mask, dim=(2, 3), keepdim=True)
        result = weighted_sum / pixel_count
        return result


class PAFE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
    
    def forward(self, x, prompt_mask):
        if prompt_mask.shape[2:] != x.shape[2:]:
            prompt_mask = F.interpolate(
                prompt_mask, size=x.shape[2:], mode='bilinear', align_corners=False
            )
        enhanced_feat = x + (x * prompt_mask)
        out = self.linear(enhanced_feat)
        prompt_i = torch.sigmoid(out)
        return enhanced_feat, prompt_i
    
class DBFE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.masked_pool = MaskedAdaptiveAvgPool2d()

        self.linear = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
    
    def forward(self, x, prompt_mask):
        # bg enhance
        bg_mask = 1 - prompt_mask
        bg_masked_x = x * bg_mask  # [B, C, H, W]
        # bg_masked_x = self.feature_processor(bg_masked_x) # [B, C, H, W]
        bg_token = self.masked_pool(bg_masked_x, bg_mask) # [B, C, 1, 1]
        bg_expanded = bg_token.expand(-1, -1, x.size(2), x.size(3)) # [B, C, H, W]
        background_suppression = x - bg_expanded # [B, C, H, W]
        # if GT_prompt is not None:
        background_suppression = background_suppression * prompt_mask
        output = x + background_suppression
        prompt_i = self.linear(output)
        prompt_i = torch.sigmoid(prompt_i)
        return output, prompt_i

class SAM2(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
            
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        
        stage_channels = [144, 288, 576, 1152]
        self.pafe = nn.ModuleList([
            PAFE(in_channels=ch) for ch in stage_channels
        ])
        
        self.dbfe = nn.ModuleList([
            DBFE(in_channels=ch) for ch in stage_channels
        ])

        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(1, dtype=torch.float32, device='cuda')) for ch in stage_channels
        ])
        self.beta = nn.ParameterList([
            nn.Parameter(torch.tensor(1, dtype=torch.float32, device='cuda')) for ch in stage_channels
        ])


    def forward(self, x, prompt_mask=None, prompt_mode=None, neg_prompt=None, name=None):
        prompt_is_none = False
        # Initialize prompt mask as zeros if not provided
        if prompt_mask is None:
            prompt_mask = torch.zeros(
                (x.shape[0], 1, x.shape[2], x.shape[3]), 
                dtype=x.dtype, device=x.device
            )
            prompt_is_none = True
        elif prompt_mask.dim() == 3:
            prompt_mask = prompt_mask.unsqueeze(1).float()
            
        # Process spatial prompts (Box/Scribble/OBB)
        if prompt_mode in ['Box','Scribble','OBB']:
            np_prompt = to_uint8_binary(prompt_mask[:,0])
            coords_list, prompt_mask = get_coords_and_obb(np_prompt, prompt_mode)
            
        # Initial feature transformation: patch embedding + positional encoding
        x = self.encoder.patch_embed(x)
        x = x + self.encoder._get_pos_embed(x.shape[1:3])
        outputs = [] # Stores intermediate feature maps
        GT_prompt = prompt_mask # Save initial prompts for loss calculation
        stage_idx, prompt_loss = 0, 0 # Initialize stage counter and loss accumulator
        # Process through encoder blocks
        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x) 
            # Process at stage endpoints
            if i in self.encoder.stage_ends:
                x_feat = x.permute(0, 3, 1, 2) # Rearrange features to [B, C, H, W] format
                # PAFE
                pafe_feat, prompt_i = self.pafe[stage_idx](x_feat, prompt_mask)
                prompt_i = F.interpolate(prompt_i, size=pafe_feat.shape[2:], mode='bilinear', align_corners=False)
                # DBFE
                dbfe_feat, prompt_i = self.dbfe[stage_idx](pafe_feat, prompt_i)
                # Weighted feature enhancement
                enhanced_feat = self.alpha[stage_idx] * pafe_feat + self.beta[stage_idx] * dbfe_feat
                # Prepare prompt for next stage and loss calculation
                prompt_i = F.interpolate(prompt_i, size=GT_prompt.shape[2:], mode='bilinear', align_corners=False)
                # Prompt Supervision
                if prompt_mode in ['OBB','Box','Scribble']:
                    prompt_i_M2O = prompt_M2O(prompt_i, coords_list)
                    prompt_loss += get_prompt_loss(prompt_i_M2O, GT_prompt, neg_prompt, prompt_mode)
                elif prompt_mode in ['Point','Circle']:
                    prompt_loss += get_prompt_loss(prompt_i, GT_prompt, neg_prompt, prompt_mode)
                # Update prompt mask for next stage
                prompt_mask = prompt_i
                # Prepare features for next transformer block
                x = enhanced_feat.permute(0, 2, 3, 1)
                outputs.append(enhanced_feat)
                stage_idx += 1

        return outputs, prompt_loss / (stage_idx+1)
    
    
    def initialize(self, checkpoint_path=None):
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            build_sam2(model_cfg, checkpoint_path)
        else:
            build_sam2(model_cfg)