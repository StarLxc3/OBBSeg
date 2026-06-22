import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
import os
import sys
import numpy as np
import argparse
sys.dont_write_bytecode = True
sys.path.insert(0, '../')
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from model import OBBSeg
from dataset import TestData
from medpy.metric.binary import hd95

class Test(object):
    def __init__(self, cfg):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        ## dataset
        self.cfg       = cfg 
        self.data      = TestData(cfg)
        self.loader    = DataLoader(dataset=self.data, batch_size=cfg.batch_size, shuffle=False, num_workers=int(self.cfg.num_workers), pin_memory=True, collate_fn=self.data.collect_fn)
        self.model     = OBBSeg(cfg).cuda()
        self.model.eval()
        
    def test_prediction(self):
        with torch.no_grad():
            mae, iou, dice, hd95_total, cnt = 0, 0, 0, 0, 0
            self.start_event.record()
            for image, mask, obb_coords, prompt_list, name_list in tqdm(self.loader):  
                B, C, H, W         = mask.shape
                prompt_mask     = get_prompt_mask((W, H), (W, H), prompt_list, self.cfg.prompt_mode)
                pred, _         = self.model(image.cuda(non_blocking=True).float(), prompt_mask, self.cfg.prompt_mode, None, name_list[0])
                pred            = F.interpolate(pred, size=(H, W), mode='bilinear')
                pred            = (pred[:, 0]>0).cpu().float()
                mask            = mask[:,0]
                for idx in range(B):
                    # HD95
                    pred_np = pred[idx].cpu().numpy().astype(np.uint8)
                    mask_np = mask[idx].cpu().numpy().astype(np.uint8)
                    if np.any(pred_np) and np.any(mask_np):
                        try:
                            hd95_val = hd95(pred_np, mask_np, voxelspacing=(1, 1))
                            hd95_total += hd95_val
                        except:
                            print("HD95 calculation failed for:", name_list[idx])
                    else:
                        hd95_total += 0
                
                cnt            += B
                mae            += np.abs(pred-mask).mean()
                inter, union    = (pred*mask).sum(dim=(1,2)), (pred+mask).sum(dim=(1,2))
                iou            += ((inter+1)/(union-inter+1)).sum()
                dice           += ((2*inter+1)/(union+1)).sum()
            print('cnt=%10d | mae=%.4f | dice=%.4f | iou=%.4f | hd95=%.4f'%(cnt, mae/cnt, dice/cnt, iou/cnt, hd95_total/cnt))
            self.end_event.record()
            torch.cuda.synchronize()
            inference_time = self.start_event.elapsed_time(self.end_event)
            print(f'Inference Time: {inference_time} ms')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int,
                        default=64, help='training batch size')
    parser.add_argument('--num_worker', type=int,
                        default=4, help='decay rate of learning rate')
    parser.add_argument('--test_dataset', type=str,
                        default='PraNetDataset', help='train dataset')
    parser.add_argument('--prompt_mode', type=str,
                        default='Circle', help='class of prompt mask') # option = ['Point','Box','Scribble','Circle']
    parser.add_argument('--load_path', type=str,
                        default='SAM2')  # set the path of snapshot model
    option = parser.parse_args()
    
    prompt_mode = option.prompt_mode
    dataset = option.test_dataset
    
    class Config:
        def __init__(self, backbone, testset, prompt_mode):
            self.backbone       = backbone
            self.testset        = testset
            self.prompt_mode    = prompt_mode 
            if testset in ['SUN-SEG_Pre_Easy','SUN-SEG_Pre_Hard']:
                split = testset.split('_')
                if split[2] == 'Easy':
                    self.testset = 'SUN-SEG_Pre'
                    self.test_image = '../dataset/'+self.testset+'-Processed/TestEasyDataset/Frame'
                    self.test_mask = '../dataset/'+self.testset+'-Processed/TestEasyDataset/GT'
                else:
                    self.testset = 'SUN-SEG_Pre'
                    self.test_image = '../dataset/'+self.testset+'-Processed/TestHardDataset/Frame'
                    self.test_mask = '../dataset/'+self.testset+'-Processed/TestHardDataset/GT'
            else:
                self.test_image     = '../dataset/'+testset+'-Processed/TestDataset/Frame'
                self.test_mask      = '../dataset/'+testset+'-Processed/TestDataset/GT'
                
            ## other settingss
            self.mode           = 'test'
            self.batch_size     = option.batchsize
            self.num_workers    = option.num_worker
            self.snapshot       = option.load_path + '/model-'+self.prompt_mode+'.pt'
            self.sam2_checkpoint_path = "../pretrain/sam2_hiera_large.pt"
        
    Test(Config('SAM2', dataset, prompt_mode)).test_prediction()    
