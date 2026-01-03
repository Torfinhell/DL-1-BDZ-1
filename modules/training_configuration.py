from .config import Config
from torch import nn
import torch
def get_opt_sch(config:Config, model):
    if(config.OPTIMIZER=="AdamW"):
        optimizer=torch.optim.AdamW(model.parameters() if config.LOSS !="ArcMargin" else list(model.parameters()) + list(loss_fn.parameters()), lr=config.LEARNING_RATE)
    elif(config.OPTIMIZER=="SGD"):
        optimizer=torch.optim.SGD(model.parameters() if config.LOSS !="ArcMargin" else list(model.parameters()) + list(loss_fn.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, nesterov=True, momentum=config.MOMENTUM)
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