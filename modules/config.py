import os
import torch
class Config:
    WINDOW_SIZE:tuple[int, int]=(40, 40)
    LAST_LINEAR_SIZE:int=8400
    BATCH_SIZE:int=1024
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MEAN=None
    STD=None
    LEARNING_RATE:float=0.006615
    ACCUM_STEP:int=1
    NUM_WORKERS:int=os.cpu_count() or 1
    LOG_STEP:int=5
    NUM_EPOCHS:int=1000
    LOSS:str="ArcMargin"
    MODEL:str="RESNET18"
    NUM_CLASSES:int=200
    MARGIN_ARCFACE:int=0.02477
    SCALE_ARCFACE:int=3
    WANDB_TOKEN:str=None
    WANDB_PROJECT:str="DL-BDZ-1_exp"
    RUN_NAME:str="resnet_50_lr_8e-3_with_swa_onecycle_sgd"
    OPTIMIZER:str="SGD"
    MOMENTUM:float = 0.9
    WEIGHT_DECAY:float=0.004603
    # NUM_BLOCKS:int=3
    # DROPOUT:float=0.5
    TRAININ_DIR:str|None=None
    DATAPARALLEL:bool=False
    PCT_START:float=0.1
    SCHEDULER:str="CosineAnealing"
    # SCHEDULER:str="OneCycle"
    STEPS_PER_EPOCH:int|None=None
    CLIP_GRAD_NORM:float=5.0
    SWA_START:int|None=900
    SWA_LR:float|None=1e-2
    MAGNITUDE:int=10
    NUM_OPS_AUGS:int=1
    STOP_EPOCH:int=None

    