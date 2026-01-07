import os
import torch
class Config:
    WINDOW_SIZE:tuple[int, int]=(40, 40)
    LAST_LINEAR_SIZE:int=1500
    BATCH_SIZE:int=2048
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MEAN=None
    STD=None
    LEARNING_RATE:float=3e-2
    ACCUM_STEP:int=1
    NUM_WORKERS:int=os.cpu_count() or 1
    LOG_STEP:int=5
    NUM_EPOCHS:int=450
    LOSS:str="CE"
    MODEL:str="RESNET50"
    NUM_CLASSES:int=200
    MARGIN_ARCFACE:int=0.20
    SCALE_ARCFACE:int=16
    WANDB_TOKEN:str=None
    WANDB_PROJECT:str="DL-BDZ-1_exp"
    RUN_NAME:str="resnet_50_lr_3e-2_with_swa_onecycle_adam"
    OPTIMIZER:str="SGD"
    MOMENTUM:float = 0.9
    WEIGHT_DECAY:float=3e-3
    NUM_BLOCKS:int=3
    DROPOUT:float=0.5
    TRAININ_DIR:str|None=None
    DATAPARALLEL:bool=False
    PCT_START:float=0.1
    # SCHEDULER:str="CosineAnealing"
    SCHEDULER:str="Adam"
    STEPS_PER_EPOCH:int|None=None
    CLIP_GRAD_NORM:float=5.0
    SWA_START:int|None=350
    SWA_LR:float|None=1e-2
    MAGNITUDE:int=20
    NUM_OPS_AUGS:int=2

    