import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")
import os
import sys
import logging
import numpy as np
import argparse
from datetime import datetime

sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import torch
import torch.distributed as dist
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from model import OBBSeg
from utils import *
from preprocess import preprocess
from dataset import TestData, TrainData

rng = np.random.default_rng(seed=134)

DATASET_ALIASES = {
    'FSPD-Dataset': 'PraNetDataset',
}


def resolve_existing_dataset(dataset):
    processed_path = '../dataset/'+dataset+'-Processed'
    if os.path.exists(processed_path):
        return dataset
    alias = DATASET_ALIASES.get(dataset)
    if alias and os.path.exists('../dataset/'+alias+'-Processed'):
        return alias
    return dataset

if os.environ.get('DETECT_ANOMALY') == '1':
    torch.autograd.set_detect_anomaly(True)

def init_distributed():
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
    return distributed, local_rank, rank, world_size

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

class Train:
    def __init__(self, cfg):
        ## parameter
        self.cfg        = cfg
        self.device     = torch.device('cuda', cfg.local_rank)
        self.logger     = SummaryWriter(cfg.log_path) if cfg.is_main else None
        if cfg.is_main:
            logging.basicConfig(level=logging.INFO, filename='./train.log', filemode='a', format='[%(asctime)s | %(message)s]', datefmt='%I:%M:%S')
            logging.info('backbone=%s | dataset=%s | prompt_mode=%s'%(self.cfg.backbone,self.cfg.using_dataset,self.cfg.prompt_mode))
        ## model
        self.model      = OBBSeg(cfg).to(self.device)
        if cfg.distributed:
            self.model  = DDP(
                self.model,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
        self.model.train()
        ## data
        self.data       = TrainData(cfg)
        self.sampler    = DistributedSampler(self.data, num_replicas=cfg.world_size, rank=cfg.rank, shuffle=True) if cfg.distributed else None
        self.loader = DataLoader(
            dataset=self.data,
            batch_size=cfg.batch_size,
            shuffle=self.sampler is None,
            sampler=self.sampler,
            num_workers=int(cfg.num_workers),
            pin_memory=True,
            collate_fn=self.data.collect_fn
        )
        ## optimizer
        self.optimizer = opt.AdamW([{"params":self.model.parameters(), "initia_lr": cfg.lr}], lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.max_dice   = 0


    def forward(self):
        global_step    = 0
        scaler         = torch.cuda.amp.GradScaler()
        for epoch in range(self.cfg.epoch): 
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for i, (image, mask, obb_coords, prompt_list) in enumerate(self.loader):
                with torch.cuda.amp.autocast():
                    
                    image = image.to(self.device, non_blocking=True)
                    mask = mask.to(self.device, non_blocking=True)
                    B, C, H, W = image.shape
                    # pred 1
                    size1                = rng.choice([192, 224, 256, 288, 320, 352, 384, 416, 448])
                    image1               = F.interpolate(image, size=size1, mode='bilinear')
                    
                    neg_prompt1, neg_prompt2 = None, None
                    if self.cfg.prompt_mode in ['Scribble','Point']:
                        neg_prompt1 = F.interpolate(mask, size=size1, mode='bilinear')
                        
                    prompt_1             = get_prompt_mask((W, H), (size1, size1), prompt_list, self.cfg.prompt_mode)
                    pred1, prompt_loss1  = self.model(image1, prompt_1, self.cfg.prompt_mode, neg_prompt1)
                    pred1               = F.interpolate(pred1, size=352, mode='bilinear')
                    prompt_1            = F.interpolate(prompt_1, size=352, mode='bilinear')
                    
                    size2               = rng.choice([192, 224, 256, 288, 320, 352, 384, 416, 448])
                    while size1 == size2:
                        size2           = rng.choice([192, 224, 256, 288, 320, 352, 384, 416, 448])
                    image2              = F.interpolate(image, size=size2, mode='bilinear')
                    
                    if self.cfg.prompt_mode in ['Scribble','Point']:
                        neg_prompt2 = F.interpolate(mask, size=size2, mode='bilinear')
                    
                    prompt_2            = get_prompt_mask((W, H), (size2, size2), prompt_list, self.cfg.prompt_mode)
                    pred2, prompt_loss2 = self.model(image2, prompt_2, self.cfg.prompt_mode, neg_prompt2)
                    pred2               = F.interpolate(pred2, size=352, mode='bilinear')
                    prompt_2            = F.interpolate(prompt_2, size=352, mode='bilinear')
                    
                    loss_sc       = SC_Loss(torch.sigmoid(pred1), torch.sigmoid(pred2), mask)
                    loss_prompt    = (prompt_loss1 + prompt_loss2).mean()
                    
                    pred           = torch.cat([pred1, pred2], dim=0)
                    mask           = torch.cat([mask, mask], dim=0)
                    prompt         = torch.cat([prompt_1, prompt_2], dim=0)
                
                    if self.cfg.prompt_mode in ['Box','Scribble']:
                        np_prompt = to_uint8_binary(prompt[:,0])
                        prompt_coords_list, prompt = get_coords_and_obb(np_prompt) # OBB_prompt
                        pred_prompt      = M2O(pred, prompt_coords_list)
                    if self.cfg.prompt_mode in ['Point','Circle']:
                        pred_prompt = pred
                    
                    obb_coords = obb_coords + obb_coords
                    pred           = M2O(pred, obb_coords)
                    
                    if self.cfg.prompt_mode in ['OBB']:
                        pred_prompt = pred
                    
                    pred, mask, pred_prompt, prompt     = pred[:,0], mask[:,0], pred_prompt[:,0], prompt[:,0]
                    loss_strcution = strcution_loss(pred, mask)
                    loss_prompt += get_bce_by_prompt(pred_prompt, prompt, self.cfg.prompt_mode)
                    loss_prompt = 0.3*loss_prompt
                                    
                    loss           = loss_strcution + loss_prompt + loss_sc
                    
                ## backward
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                clip_gradient(self.optimizer, self.cfg.clip)
                scaler.step(self.optimizer)
                scaler.update()

                global_step += 1
                if self.cfg.is_main:
                    self.logger.add_scalar('lr'  , self.optimizer.param_groups[0]['lr'], global_step=global_step)
                    self.logger.add_scalars('loss', {'strcution':loss_strcution.item(), 'esc':loss_sc.item(), 'prompt':loss_prompt.item()}, global_step=global_step)
                ## print loss
                if self.cfg.is_main and global_step % 20 == 0:
                    print('{} epoch={:03d}/{:03d}, step={:04d}/{:04d}, loss_strcution={:0.4f}, loss_sc={:0.4f}, loss_prompt={:0.4f}'.format(datetime.now(), epoch, self.cfg.epoch, i, len(self.loader), loss_strcution.item(), loss_sc.item(), loss_prompt.item()))
            if self.cfg.distributed:
                dist.barrier()
            if self.cfg.is_main:
                self.evaluate(epoch)
            if self.cfg.distributed:
                dist.barrier()

    def evaluate(self, epoch):
        self.model.eval()
        model = self.model.module if hasattr(self.model, 'module') else self.model
        with torch.no_grad():
            data                = TestData(self.cfg)
            loader              = DataLoader(dataset=data, batch_size=64, shuffle=False, num_workers=int(self.cfg.num_workers), pin_memory=True, collate_fn=data.collect_fn)
            dice, iou, cnt      = 0, 0, 0
            for image, mask, obb_coords, prompt_list, name_list in loader:
                image = image.to(self.device, non_blocking=True).float()
                mask = mask.to(self.device, non_blocking=True).float()
                B, C, H, W         = mask.shape
                prompt_mask     = get_prompt_mask((W, H), (W, H), prompt_list, self.cfg.prompt_mode)
                pred, _         = model(image, prompt_mask, self.cfg.prompt_mode)
                pred            = F.interpolate(pred, size=(H, W), mode='bilinear')
                pred            = pred[:, 0] > 0
                mask            = mask[:,0]
                inter, union    = (pred*mask).sum(dim=(1,2)), (pred+mask).sum(dim=(1,2))
                dice           += ((2*inter+1)/(union+1)).sum().cpu().numpy()
                iou            += ((inter+1)/(union-inter+1)).sum().cpu().numpy()
                cnt            += B
            logging.info('epoch=%-8d | dice=%.4f | iou=%.4f | path=%s'%(epoch, dice/cnt, iou/cnt, self.cfg.test_image))

        if dice/cnt>self.max_dice:
            self.max_dice = dice/cnt
            model = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model.state_dict(), self.cfg.snapshot +'/model-'+self.cfg.prompt_mode+'.pt')
        self.model.train()

if __name__=='__main__':
    # Our model have to use at less 48 gb memory, pls make sure have enough config
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=50, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='decay rate of learning rate')
    parser.add_argument('--clip', type=float,
                        default=0.5)
    parser.add_argument('--num_worker', type=int,
                        default=4, help='decay rate of learning rate')
    parser.add_argument('--train_dataset', type=str,
                        default='FSPD-Dataset', help='train dataset')
    parser.add_argument('--prompt_mode', type=str,
                        default='Point', help='class of prompt mask') # option = ['Point','Box','Scribble','Circle']
    parser.add_argument('--save_path', type=str,
                        default='SAM2')
    option = parser.parse_args()
    distributed, local_rank, rank, world_size = init_distributed()

    prompt_mode = option.prompt_mode
    dataset = resolve_existing_dataset(option.train_dataset)
    if not os.path.exists('../dataset/'+dataset+'-Processed'):
    # if os.path.exists('../dataset/'+dataset+'-Processed'):
        if rank == 0:
            if dataset in ['SUN-SEG_Pre']:
                preprocess('../dataset/'+dataset+'/TrainDataset', dataset)
                preprocess('../dataset/'+dataset+'/TestEasyDataset', dataset)
                preprocess('../dataset/'+dataset+'/TestHardDataset', dataset)
                preprocess('../dataset/'+dataset+'/TestDataset', dataset)
                
            elif dataset in ['ISIC2018_Pre']:
                preprocess('../dataset/'+dataset+'/TrainDataset', dataset)
                preprocess('../dataset/'+dataset+'/TestDataset', dataset)
                preprocess('../dataset/'+dataset+'/ValidDataset', dataset)
            else:
                preprocess('../dataset/'+dataset+'/TrainDataset', dataset)
                preprocess('../dataset/'+dataset+'/TestDataset', dataset)
    if distributed:
        dist.barrier()

    ## hyperparameter config
    class Config:
        def __init__(self, backbone, using_dataset, prompt_mode):
            ## set the backbone type
            self.backbone       = backbone
            self.using_dataset  = dataset
            ## set the path of training dataset
            self.train_image = '../dataset/'+using_dataset+'-Processed/TrainDataset/Frame'
            self.train_mask = '../dataset/'+using_dataset+'-Processed/TrainDataset/OBB'
            self.prompt_mode = prompt_mode
            
            if using_dataset in ['ISIC2018_Pre']:
                self.test_image = '../dataset/'+using_dataset+'-Processed/ValidDataset/Frame'
                self.test_mask = '../dataset/'+using_dataset+'-Processed/ValidDataset/GT'
            else:
                self.test_image = '../dataset/'+using_dataset+'-Processed/TestDataset/Frame'
                self.test_mask = '../dataset/'+using_dataset+'-Processed/TestDataset/GT'

            ## set the path of logging and saving
            self.log_path       = self.backbone+"/"+self.using_dataset+'/log'
            self.sam2_checkpoint_path = "../pretrain/sam2_hiera_large.pt"
            self.snapshot = option.save_path+"/"+self.using_dataset
            os.makedirs(self.log_path, exist_ok=True)
            
            self.mode           = 'train'
            self.epoch          = option.epoch
            self.batch_size     = option.batchsize
            self.lr             = option.lr
            self.num_workers    = option.num_worker
            self.weight_decay   = option.weight_decay
            self.clip           = option.clip
            self.distributed    = distributed
            self.local_rank     = local_rank
            self.rank           = rank
            self.world_size     = world_size
            self.is_main        = rank == 0

    ## training    
    try:
        Train(Config('SAM2', dataset, prompt_mode)).forward()
    finally:
        cleanup_distributed()
