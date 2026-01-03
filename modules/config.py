import os
import torch
class Config:
    WINDOW_SIZE=(40, 40)
    LAST_LINEAR_SIZE=1000
    BATCH_SIZE=2048
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MEAN=None
    STD=None
    ROTATE_LIMIT=45
    SCALE_LIMIT=0.1
    SHIFT_LIMIT=0.1
    LEARNING_RATE=8e-3
    ACCUM_STEP=1
    NUM_WORKERS=os.cpu_count() or 1
    LOG_STEP=5
    NUM_EPOCHS=1500
    LOSS="CE"
    MODEL="RESNET18"
    NUM_CLASSES=200
    MARGIN_ARCFACE=0.20
    SCALE_ARCFACE=16
    WANDB_TOKEN=None
    WANDB_PROJECT="DL-BDZ-1_exp"
    RUN_NAME="first_run"
    OPTIMIZER="SGD"
    MOMENTUM = 0.9
    WEIGHT_DECAY=3e-3
    NUM_BLOCKS=3
    DROPOUT=0.5
    TRAININ_DIR=None
    DATAPARALLEL=False
    PCT_START=0.1
    SCHEDULER="CosineAnealing"
    STEPS_PER_EPOCH=None