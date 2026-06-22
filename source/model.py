import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from SAM2 import SAM2
from utils import weight_init

class Fusion(nn.Module):
    def __init__(self, channels):
        super(Fusion, self).__init__()
        self.linear2 = nn.Sequential(nn.Conv2d(channels[1], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear3 = nn.Sequential(nn.Conv2d(channels[2], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear4 = nn.Sequential(nn.Conv2d(channels[3], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))

    def forward(self, x1, x2, x3, x4):
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        x3 = x3.contiguous()
        x4 = x4.contiguous()
        
        x2, x3, x4   = self.linear2(x2), self.linear3(x3), self.linear4(x4)
        x4           = F.interpolate(x4, size=x2.size()[2:], mode='bilinear')
        x3           = F.interpolate(x3, size=x2.size()[2:], mode='bilinear')
        out          = x2*x3*x4
        return out

    def initialize(self):
        weight_init(self)


class OBBSeg(nn.Module):
    def __init__(self, cfg):
        super(OBBSeg, self).__init__()
        if cfg.backbone == 'SAM2':
            # citation: 
            # @misc{ravi2024sam2segmentimages,
            #       title={SAM 2: Segment Anything in Images and Videos}, 
            #       author={Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman Rädle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Dollár and Christoph Feichtenhofer},
            #       year={2024},
            #       eprint={2408.00714},
            #       archivePrefix={arXiv},
            #       primaryClass={cs.CV},
            #       url={https://arxiv.org/abs/2408.00714}, 
            # }
            self.backbone = SAM2(cfg.sam2_checkpoint_path)            
            channels      = [144, 288, 576, 1152]
        
        self.fusion = Fusion(channels)
        self.linear = nn.Conv2d(64, 1, kernel_size=1)
        
        ## initialize
        if cfg.mode=='train':
            weight_init(self)
        elif cfg.mode=='test':
            checkpoint = torch.load(cfg.snapshot, map_location='cpu')
            state_dict = checkpoint.module if hasattr(checkpoint, 'module') else checkpoint
            if isinstance(state_dict, OBBSeg):
                state_dict = state_dict.state_dict()
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError
        
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init(m)
        
    def forward(self, x, prompt=None, prompt_mode=None, neg_prompt=None, name=None):
        features, prompt_loss = self.backbone(x, prompt, prompt_mode, neg_prompt, name)
        x1, x2, x3, x4 = features
        pred = self.fusion(x1, x2, x3, x4)
        pred = self.linear(pred)
        
        return pred, prompt_loss
    
