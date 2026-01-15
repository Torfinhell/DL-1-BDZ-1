from .config import Config
from torch import nn
import torch
from torch.utils import data 
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random 
def get_opt_sch(config:Config, model):
    if(config.OPTIMIZER=="AdamW"):
        optimizer=torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    elif(config.OPTIMIZER=="SGD"):
        optimizer=torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, nesterov=True, momentum=config.MOMENTUM)
    else:
        raise NotImplementedError("Optimizer is not implemented")
    if config.SCHEDULER=="CosineAnealing":
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    elif(config.SCHEDULER=="OneCycle"):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.LEARNING_RATE,
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=config.STEPS_PER_EPOCH,
            pct_start=config.PCT_START
        )
    else:
        raise NotImplementedError("Scheduler is not implemented")
    return optimizer, scheduler
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CustomBatchSampler(Sampler):
    def __init__(self, ds):
        self.ds=ds
        self.label_to_ind=defaultdict(list)
        self.ind_to_label:dict[int, int]={}
        for i, (image, label) in enumerate(self.ds):
            self.label_to_ind[label].append(i)
            self.ind_to_label[i]=label
        self.num_labels=len(self.label_to_ind)
    def __iter__(self):
        for list_ind in self.label_to_ind.values():
            random.shuffle(list_ind)
        for i in range(len(self)):
            batch=[]
            for list_ind in self.label_to_ind.values():
                batch.append(list_ind[i])
            yield batch
    def __len__(self):
        return min(len(v) for v in self.label_to_ind.values())
def get_dataloader(ds, config, part: str = "train"):
    if config.LOSS == "ArcMargin":
        return data.DataLoader(
            ds,
            batch_sampler=CustomBatchSampler(ds),
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
    else:
        return data.DataLoader(
            ds,
            batch_size=config.BATCH_SIZE,
            shuffle=(part == "train"),
            drop_last=(part == "train"),
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )