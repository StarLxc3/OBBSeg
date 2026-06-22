import os
import random
import cv2
import numpy as np
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------initation function---------------------------:
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.PReLU)):
            pass
        else:
            m.initialize()

# ---------------------------loss fuction---------------------------:
def SC_Loss(pred1, pred2, mask):
    # citation: 
    # @inproceedings{wei2023weakpolyp,
    #     title={Weakpolyp: You only look bounding box for polyp segmentation},
    #     author={Wei, Jun and Hu, Yiwen and Cui, Shuguang and Zhou, S Kevin and Li, Zhen},
    #     booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    #     pages={757--766},
    #     year={2023},
    #     organization={Springer}
    # }
    loss_sc        = (pred1-pred2).abs()
    selected       = loss_sc[mask[:,0:1]==1]
    if selected.numel() == 0:
        return loss_sc.mean() * 0
    return selected.mean()

def strcution_loss(pred, mask):
    loss_ce        = F.binary_cross_entropy_with_logits(pred, mask)
    pred           = torch.sigmoid(pred)
    inter          = (pred*mask).sum(dim=(1,2))
    union          = (pred+mask).sum(dim=(1,2))
    loss_dice      = 1-(2*inter/(union+1)).mean()
    return 2*(loss_ce + loss_dice).mean()

def get_bce_by_prompt(pred, mask, prompt_mode):
    pred = torch.sigmoid(pred)
    loss = -(mask * torch.log(pred + 1e-6) + (1 - mask) * torch.log(1 - pred + 1e-6))
    if prompt_mode in ['Circle']:
        selected = loss[mask[:]==0]
        loss = selected.mean() if selected.numel() > 0 else loss.mean() * 0
    if prompt_mode in ['Scribble','Point']:
        selected = loss[mask[:]==1]
        loss = selected.mean() if selected.numel() > 0 else loss.mean() * 0
    if prompt_mode in ['Box','OBB']:
        loss = loss.mean()
    return  loss

# ---------------------------M2O transformation implement---------------------------:
def M2O(pred, obb_coords):
    B, C, H, W = pred.shape
    # Process each image in batch
    for j in range(len(pred)):
        boxs_points = obb_coords[j]
        if not boxs_points: # Skip if no OBBs
            continue
        boxs_points_np = np.array(boxs_points, dtype=np.float32)
        num_boxes = boxs_points_np.shape[0]
        angles = []
        bg = torch.zeros([num_boxes, 1, H, W], device=pred.device) # Buffer for OBB regions
        # Process each OBB
        for k in range(num_boxes):
            pts = boxs_points_np[k]
            # Convert OBB points to integer coordinates
            ys = [int(y) for x, y in pts]
            xs = [int(x) for x, y in pts]
            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = max(0, min_y), min(H, max_y + 1)
            min_x, max_x = max(0, min_x), min(W, max_x + 1)
            if min_y >= max_y or min_x >= max_x:
                continue
            # Calculate OBB rotation angle (normalized to [-90°, 90°])
            _, (w, h), angle = cv2.minAreaRect(pts)
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            angles.append(angle)
            # Spilt
            bg[k:k+1, 0:1, min_y:max_y, min_x:max_x] = pred[j:j+1, 0:1, min_y:max_y, min_x:max_x]

        angles = torch.tensor(angles, device=pred.device)
        cos_theta = torch.cos(torch.deg2rad(angles))
        sin_theta = torch.sin(torch.deg2rad(angles))
        # Create rotation matrices
        trans_matrix = torch.zeros(num_boxes, 3, 3, device=pred.device)
        trans_matrix[:, 0, 0] = cos_theta
        trans_matrix[:, 0, 1] = -sin_theta
        trans_matrix[:, 1, 0] = sin_theta                         
        trans_matrix[:, 1, 1] = cos_theta                         
        trans_matrix[:, 2, 2] = 1
        # Rotate regions to axis-aligned
        grid = F.affine_grid(trans_matrix[:, :2, :], (num_boxes, 1, H, W), align_corners=False)
        rotated_region = F.grid_sample(bg, grid, mode='nearest', padding_mode='zeros')
        # Max projections on rotated space and fusion
        predW = rotated_region.max(dim=2, keepdim=True)[0]  
        predH = rotated_region.max(dim=3, keepdim=True)[0]  
        back_project_region = torch.minimum(predW, predH)
        # Create inverse rotation matrices
        inv_trans_matrix = torch.zeros(num_boxes, 3, 3, device=pred.device)
        inv_trans_matrix[:, 0, 0] = cos_theta 
        inv_trans_matrix[:, 0, 1] = sin_theta 
        inv_trans_matrix[:, 1, 0] = -sin_theta
        inv_trans_matrix[:, 1, 1] = cos_theta 
        inv_trans_matrix[:, 2, 2] = 1
        # Rotate refined regions back to original orientation
        inv_grid = F.affine_grid(inv_trans_matrix[:, :2, :], (num_boxes, 1, H, W), align_corners=False)
        inv_rotated_region = F.grid_sample(back_project_region, inv_grid, mode='nearest', padding_mode='zeros')
        # maximum fusion
        fusion_region = torch.max(inv_rotated_region, dim=0, keepdim=True)[0]
        region_mask = (fusion_region != 0).float()
        pred_comb = torch.where(
            region_mask.bool(),  
            torch.maximum(fusion_region[0:1], pred[j:j+1]), 
            pred[j:j+1]     
        )
        pred = pred.clone()
        pred[j:j+1] = pred_comb # Update current image in batch
    return pred

# ---------------------------others---------------------------:
def get_prompt_mask(original_size, target_size, prompt_list=None, prompt_mode = None):
    if prompt_list is None:
        return None
    if prompt_mode in ['Scribble','Circle','Point','OBB','Box']:
        prompt_masks = torch.tensor(np.array(prompt_list), dtype=torch.float32).cuda().float()
        prompt_masks = F.interpolate(prompt_masks, size=target_size[0], mode='bilinear')

    return prompt_masks

def aabb2obb(boxes_list):
    obb_boxes_list=[]
    for boxes in boxes_list:
        obb_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
            size = (width, height)
            angle = 0.0 
            rect = (center, size, angle)
            obb_points = cv2.boxPoints(rect)
            obb_points =obb_points.astype(np.int32).tolist()
            obb_boxes.append(obb_points)
        obb_boxes_list.append(obb_boxes)
    return obb_boxes_list

def to_uint8_binary(mask):
    mask_np = mask.detach().cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    return mask_np

def get_coords_and_obb(np_prompt, prompt_mode=None):
    coords_list = []
    prompt_obb = np.zeros_like(np_prompt, dtype=np.uint8)
    for k in range(np_prompt.shape[0]):
        contours = cv2.findContours(np_prompt[k], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        coords = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            (cx, cy), (width, height), angle = rect
            if prompt_mode != None and prompt_mode == 'OBB':
                rect = (cx, cy), (width + 10, height + 10), angle
            obb_points = cv2.boxPoints(rect)
            obb_points = obb_points.astype(np.int32)  
            cv2.fillConvexPoly(prompt_obb[k], obb_points, color=255)
            coords.append(obb_points)
        coords_list.append(coords)
    prompt_obb = prompt_obb.astype(np.float32) / 255.0
    prompt_obb = torch.tensor(np.array(prompt_obb), dtype=torch.float32).cuda().unsqueeze(1).float()
    return coords_list, prompt_obb

def get_prompt_loss(pred, mask, neg_mask=None, prompt_mode=None):
    loss = 0.0
    if prompt_mode == None:
        return loss
    loss = -(mask * torch.log(pred + 1e-6) + (1 - mask) * torch.log(1 - pred + 1e-6))
    if prompt_mode in ['Scribble','Point']:
        if neg_mask != None:
            pos = loss[mask[:,0:1]==1]
            neg = loss[neg_mask[:,0:1]==0]
            pos_loss = pos.mean() if pos.numel() > 0 else loss.mean() * 0
            neg_loss = neg.mean() if neg.numel() > 0 else loss.mean() * 0
            loss = pos_loss + neg_loss
        else:
            pos = loss[mask[:,0:1]==1]
            loss = pos.mean() if pos.numel() > 0 else loss.mean() * 0
    return loss.mean()

def prompt_M2O(stage_prompt, coords_list=None, prompt_mode=None):
    pred = torch.cat([stage_prompt],dim=0)
    pred = M2O(pred, coords_list)
    return pred

# ---------------------------image saving function---------------------------:
def save_image(image, name_prefix, j, pts=None, boxes=None, use_heatmap=False):
    os.makedirs('./tempsave', exist_ok=True)
    if torch.is_tensor(image):
        img_tensor = image.squeeze().detach().cpu()
        
        min_val = img_tensor.min().item()
        max_val = img_tensor.max().item()
    
        if min_val >= -1.0 and min_val < 0 and max_val <= 1.0:
            img_np = (img_tensor.numpy() * 0.5 + 0.5) * 255  # tanh→[0,255]
        elif min_val >= 0 and max_val <= 1:
            img_np = img_tensor.numpy() * 255                # sigmoid→[0,255]
        else:
            img_np = (img_tensor.numpy() - min_val) / (max_val - min_val) * 255 
    else:
        img_np = np.asarray(image).squeeze()
        min_val = img_np.min().item() if hasattr(img_np, 'min') else np.min(img_np)
        max_val = img_np.max().item() if hasattr(img_np, 'max') else np.max(img_np)
        if min_val >= -1.0 and min_val < 0 and max_val <= 1.0:
            img_np = (img_np * 0.5 + 0.5) * 255.0
        elif min_val >= 0 and max_val <= 1:
            img_np = img_np * 255.0
        else:
            if max_val - min_val < 1e-5:
                img_np = np.ones_like(img_np) * 128   
            else:
                img_np = (img_np - min_val) / (max_val - min_val) * 255.0
                
    img_np = img_np.clip(0, 255).astype(np.uint8)

    if use_heatmap:
        heatmap = cv2.applyColorMap(img_np, cv2.COLORMAP_JET)
        img_np = heatmap
    
    if len(img_np.shape) == 3 and img_np.shape[0] == 3:  
        img_np = img_np.transpose(1, 2, 0)               

    if img_np.ndim == 2: 
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:  
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else: 
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    if pts is not None:
        pts = np.array(pts, dtype=np.float32)
        rect = cv2.minAreaRect(pts)  
        box = cv2.boxPoints(rect)  
        box = box.astype(np.int32)  
        cv2.polylines(img_np, [box], isClosed=True, color=(0, 255, 0), thickness=2)

    if boxes is not None:
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), color=(255, 255, 255), thickness=2)
    
    filename = f'./tempsave/{name_prefix}_batch{j}.png'
    cv2.imwrite(filename, img_np)
