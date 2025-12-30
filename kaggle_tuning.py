import optuna
import torch
from pathlib import Path
from main import train_detector, Config 
TRAIN_IMAGES = "/kaggle/input/bdz-dl-1/bhw1/trainval"
LABELS_CSV = "/kaggle/input/bdz-dl-1/bhw1/labels.csv"
SAVE_DIR = "optuna_models"
Path(SAVE_DIR).mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def objective(trial: optuna.Trial):
    config = Config()
    config.BATCH_SIZE = 4096
    # config.LEARNING_RATE = trial.suggest_float(
    #     "learning_rate", 1e-4,5e-3, log=True
    # )

    # config.MARGIN_ARCFACE = trial.suggest_float(
    #     "margin_arcface", 0.1,0.5
    # )

    # config.SCALE_ARCFACE = trial.suggest_int(
    #     "scale_arcface", 8, 64, step=4
    # )
    # config.LAST_LINEAR_SIZE = trial.suggest_int(
    #     "last_linear_size", 200, 1000, step=200
    # )
    # config.MODEL = trial.suggest_categorical(
    #     "model",
    #     ["RESNET18", "RESNET34", "RESNET50", "MyModel"]
    #     # ["RESNET18", "RESNET34", "RESNET50", "MNASNET0_5"]
    # )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.WANDB_TOKEN = "00a0bbd0a1ced8fae98a5550e703cbd7a912eb84"
    config.RUN_NAME=f"batch_size_{config.BATCH_SIZE}_lr_{config.LEARNING_RATE:.5f}_m_{config.MARGIN_ARCFACE:.2f}_s_{config.SCALE_ARCFACE:.0f}_model_{config.MODEL}_last_linear_size_{config.LAST_LINEAR_SIZE}"
    config.LOSS = "ArcMargin"
    config.NUM_EPOCHS = 30           
    config.DEVICE = DEVICE
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

    study.optimize(
        objective,
        n_trials=30, 
        timeout=None
    )

    print("\n🏆 Best trial:")
    trial = study.best_trial

    print(f"  Accuracy: {trial.value:.4f}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")
