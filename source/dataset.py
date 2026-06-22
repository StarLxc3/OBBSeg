import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

class TrainData(Dataset):
    def __init__(self, cfg):
        self.samples   = []
        folder = ''
        self.cfg = cfg
        for name in os.listdir(os.path.join(cfg.train_image, folder)):
            image = cfg.train_image + '/' + folder + name
            mask = cfg.train_mask + '/' + folder + name.replace('.jpg', '.png')
            coord = image.replace("Frame", "Coord").replace(".jpg", ".txt").replace(".png", ".txt")
            if self.cfg.prompt_mode in ['Circle','Scribble','Point','OBB','Box']:
                prompt = image.replace("Frame", cfg.prompt_mode).replace(".jpg", ".png")
            self.samples.append((image, mask, coord, prompt))
            
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        ,bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]),seed=137,
        additional_targets={
            'prompt_mask': 'mask'  
        })

    def __getitem__(self, idx):
        image_path, mask_path, coord_path, prompt_path = self.samples[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.float32(mask > 128)[..., None] 
        prompt_mask = None
        
        if self.cfg.prompt_mode in ['Circle','Scribble','Point','OBB','Box']:
            prompt = cv2.imread(prompt_path, cv2.IMREAD_GRAYSCALE)
            prompt_mask = np.float32(prompt > 128)[..., None]
            
        coord, cnt = self.coord_loader(coord_path)
        
        # augment
        keypoints = [pt for box in coord for pt in box]
        if self.cfg.prompt_mode in ['Circle','Scribble','Point','OBB','Box']:
            augmented = self.transform(image=image, mask=mask, keypoints=keypoints, prompt_mask=prompt_mask)
            
        augmented_keypoints = augmented['keypoints']
        obb_points = []
        current_idx = 0
        for count in cnt:
            current_box = augmented_keypoints[current_idx : current_idx + count]
            obb_points.append(current_box)
            current_idx += count

        if self.cfg.prompt_mode in ['Circle','Scribble','Point','OBB','Box']:
            prompt = augmented['prompt_mask'].permute(2, 0, 1)
        
        return augmented['image'], augmented['mask'].permute(2, 0, 1), obb_points, prompt
    
    def collect_fn(self, items):
        images, masks, obb_coords, prompt_list = [], [], [], []
        for item in items:
            image, mask, obb, prompt = item
            images.append(image)
            masks.append(mask)
            obb_coords.append(obb)
            prompt_list.append(prompt)
        return torch.stack(images), torch.stack(masks), obb_coords, prompt_list
    
    
    def boxs_loader(self, path):
        boxes = []
        with open(path, 'r') as f:
            boxes = f.readlines()
        boxes = [list(map(int, map(float, box.strip().split(',')))) for box in boxes]
        return boxes
    
    def coord_loader(self, path):
        obbs = []
        cnt = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    points = line.split(';')
                    box = []
                    for pt in points:
                        x, y = map(int, pt.split(','))
                        box.append([x, y])
                    obbs.append(box)
                    cnt.append(len(box))
        return obbs, cnt

    def __len__(self):
        return len(self.samples)


class TestData(Dataset):
    def __init__(self, cfg):
        self.samples  = []
        self.cfg      = cfg
        folder = ''
        for name in os.listdir(os.path.join(cfg.test_image, folder)):
            image = cfg.test_image + '/' + folder + name
            mask = cfg.test_mask + '/' + folder + name.replace('.jpg', '.png')
            coord = image.replace("Frame", "Coord").replace(".jpg", ".txt").replace(".png", ".txt")
            if self.cfg.prompt_mode in ['Circle','Scribble','Point','OBB','Box']:
                prompt = image.replace("Frame", cfg.prompt_mode).replace(".jpg", ".png")
            self.samples.append((image, mask, coord, prompt, name))

        print('Test Data: %s,   Test Samples: %s'%(cfg.test_image, len(self.samples)))

        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(320, 320),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        ,bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]),seed=137,
        additional_targets={
            'prompt_mask': 'mask'  
        })

    def __getitem__(self, idx):
        image_path, mask_path, coord_path, prompt_path, name = self.samples[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.float32(mask > 128)[..., None]
        coord, cnt = self.coord_loader(coord_path)
        prompt_mask = None
        
        if self.cfg.prompt_mode in ['Circle','Scribble','Point','OBB','Box']:
            prompt = cv2.imread(prompt_path, cv2.IMREAD_GRAYSCALE)
            prompt_mask = np.float32(prompt > 128)[..., None]
            
        keypoints = [pt for box in coord for pt in box]
        if self.cfg.prompt_mode in ['Circle','Scribble','Point','OBB','Box']:
            augmented = self.transform(image=image, mask=mask, keypoints=keypoints, prompt_mask=prompt_mask)
            
        augmented_keypoints = augmented['keypoints']
        obb_points = []
        current_idx = 0
        for count in cnt:
            current_box = augmented_keypoints[current_idx : current_idx + count]
            obb_points.append(current_box)
            current_idx += count
            
        if self.cfg.prompt_mode in ['Circle','Scribble','Point','OBB','Box']:
            prompt = augmented['prompt_mask'].permute(2, 0, 1)
            
        return augmented['image'], augmented['mask'].permute(2, 0, 1), obb_points, prompt, name

    def collect_fn(self, items):
        images, masks, obb_coords, prompt_list, name_list = [], [], [], [], []
        for item in items:
            image, mask, obb, prompt, name = item
            images.append(image)
            masks.append(mask)
            obb_coords.append(obb)
            prompt_list.append(prompt)
            name_list.append(name)
        return torch.stack(images), torch.stack(masks), obb_coords, prompt_list, name_list
    
    
    def boxs_loader(self, path):
        boxes = []
        with open(path, 'r') as f:
            boxes = f.readlines()
        boxes = [list(map(int, map(float, box.strip().split(',')))) for box in boxes]
        return boxes
    
    def coord_loader(self, path):
        obbs = []
        cnt = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    points = line.split(';')
                    box = []
                    for pt in points:
                        x, y = map(int, pt.split(','))
                        box.append([x, y])
                    obbs.append(box)
                    cnt.append(len(box))
        return obbs, cnt

    def __len__(self):
        return len(self.samples)