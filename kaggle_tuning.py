import optuna
import torch
from pathlib import Path
from main import train_detector 
import argparse
from modules.config import Config
TRAIN_IMAGES = "/kaggle/input/bhw1/trainval"
LABELS_CSV = "/kaggle/input/bhw1/labels.csv"
SAVE_DIR = "optuna_models"
WANDB_TOKEN=None
Path(SAVE_DIR).mkdir(exist_ok=True)
# WANDB_TOKEN=None
# TRAIN_IMAGES = "bhw-1-dl-2025-2026/bhw1/trainval"
# LABELS_CSV = "bhw-1-dl-2025-2026/bhw1/labels.csv"
# SAVE_DIR = "optuna_models"
def objective(trial: optuna.Trial):
    config = Config()
    config.LEARNING_RATE = trial.suggest_float(
        "learning_rate", 1e-5,1e-2, log=True
    )
    config.WEIGHT_DECAY = trial.suggest_float(
        "weight_decay", 1e-5,1e-2, log=True
    )
    config.MAGNITUDE=trial.suggest_int(
        "magnitude", 10, 70, step=10
    )
    # size=trial.suggest_int(
    #     "window_size", 20, 60, step=4
    # )
    # config.WINDOW_SIZE=(size, size)
    config.MODEL = trial.suggest_categorical(
        "model",
        ["RESNET18", "RESNET34", "RESNET50"]
    )
    # config.MODEL="RESNET18"
    # config.NUM_BLOCKS=trial.suggest_int(
    #     "num_blocks",1, 5, step=1
    # )
    config.LAST_LINEAR_SIZE = trial.suggest_int(
        "last_linear_size", 200, 10000, step=200
    )
    config.MARGIN_ARCFACE = trial.suggest_float(
        "margin_arcface", 0.0,0.5
    )

    config.SCALE_ARCFACE = trial.suggest_int(
        "scale_arcface", 3, 30, step=4
    )
    
    config.BATCH_SIZE = 2048
    config.SWA_START=None
    # config.LOSS = "ArcMargin"
    # config.OPTIMIZER="SGD"
    config.NUM_EPOCHS = 450
    config.STOP_EPOCH=10           
    config.WANDB_TOKEN = WANDB_TOKEN
    config.RUN_NAME=f"model_{config.MODEL}_arcface_m_{config.MARGIN_ARCFACE:.2f}"
    # config.RUN_NAME=f"resnet50_ablation_lr_{config.LEARNING_RATE:.6f}_wd_{config.WEIGHT_DECAY:.6f}_m_{config.MAGNITUDE}"
    config.WANDB_PROJECT="Tuning Arcface"
    config.TRAININ_DIR=TRAIN_IMAGES
    try:
        best_acc = train_detector(
            labels_csv=LABELS_CSV,
            images_path=TRAIN_IMAGES,
            config=config,
            save_model_path=None  
        )
    except RuntimeError as e:
        print("Runtime error:", e)
        return 0.0

    return best_acc


if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        study_name="arcface_model_search"
    )
    parser=argparse.ArgumentParser(description="tuning script")
    parser.add_argument("--wandb_token", type=str, help="wandb token")
    args=parser.parse_args()
    WANDB_TOKEN=args.wandb_token
    study.optimize(
        objective,
        n_trials=100, 
        timeout=None
    )

    print("\n🏆 Best trial:")
    trial = study.best_trial

    print(f"  Accuracy: {trial.value:.4f}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")
