

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import torchvision.transforms.v2 as T
from torch.utils import data 
import os
from torch.utils import data 
import argparse
import torch.nn.functional as F
import wandb
from pathlib import Path
from modules.models import get_model
from modules.losses import get_loss
from modules.training_configuration import get_opt_sch
from modules.config import Config
from modules.dataset import MyDataset
from modules.models import MyModel
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn


def grad_norm(model:MyModel):
    model_ = model.module if isinstance(model, nn.DataParallel) else model
    total_norm=0.0
    for param in model_.parameters():
        if param.grad is not None:
            total_norm+=param.grad.detach().norm(2)**2
    return total_norm**0.5

#MAIN_FUNCTIONS
def train_detector(labels_csv:str, images_path:str,config=Config(), save_model_path:str|None=None):
    if(config.WANDB_TOKEN is not None):
        wandb.login(key=config.WANDB_TOKEN)
        wandb.init(
            project=config.WANDB_PROJECT,
            name=config.RUN_NAME,
            config=vars(config),
            resume="allow"
        )
    if(config.DEVICE==torch.device("cuda:0")):
        torch.cuda.empty_cache()
    if(save_model_path is not None):
        os.makedirs(save_model_path, exist_ok=True)
    ds_train=MyDataset(images_path, labels_csv,mode="train",  config=config)
    dl_train=data.DataLoader(
        ds_train, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    ds_val=MyDataset(images_path, labels_csv,mode="valid",  config=config)
    dl_val=data.DataLoader(
        ds_val, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=config.NUM_WORKERS,
    )
    model=get_model(config)
    model, loss_fn=get_loss(model,config)
    model = model.to(config.DEVICE)
    loss_fn=loss_fn.to(config.DEVICE)
    if config.DATAPARALLEL:
        model = nn.DataParallel(model, device_ids=[0,1])
    config.STEPS_PER_EPOCH=len(dl_train)
    optimizer, scheduler=get_opt_sch(config, model)
    if(config.SWA_START is not None):
        swa_model=AveragedModel(model).to(config.DEVICE)
        swa_scheduler=SWALR(optimizer, swa_lr=config.SWA_LR, anneal_epochs=(config.NUM_EPOCHS-config.SWA_START)//10, anneal_strategy="cos")
    best_acc=0
    global_step = 0
    for e in range(config.NUM_EPOCHS):
        model.train()
        train_loss=[]
        val_loss=[]
        optimizer.zero_grad()
        pbar = tqdm(enumerate(dl_train), total=len(dl_train), desc="Training...")
        for i, (x_batch, y_batch) in pbar:
            x_batch=x_batch.to(config.DEVICE)
            y_batch=y_batch.to(config.DEVICE)
            p_batch=model(x_batch)
            loss=loss_fn(p_batch, y_batch)
            train_loss.append(loss.item())
            (loss / config.ACCUM_STEP).backward()
            if((i+1)%config.ACCUM_STEP==0):
                if config.CLIP_GRAD_NORM is not None:
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=config.CLIP_GRAD_NORM
                    )
                    post_clip_norm = grad_norm(model)
                else:
                    pre_clip_norm = grad_norm(model)
                    post_clip_norm = pre_clip_norm
                if config.WANDB_TOKEN is not None:
                    wandb.log({
                        "grad_norm/pre": pre_clip_norm.item(),
                        "grad_norm/post": post_clip_norm.item(),
                    }, step=global_step)
                optimizer.step()
                optimizer.zero_grad()
                if(config.SCHEDULER=="OneCycle"):
                    if(config.SWA_START is not None and e>=config.SWA_START):
                        swa_model.update_parameters(model)
                        swa_scheduler.step()
                    else:
                        scheduler.step()
                global_step+=1
            pbar.set_postfix(train_loss=f"{loss.item():.4f}")
        if config.SCHEDULER=="CosineAnealing":
            if(config.SWA_START is not None and e>=config.SWA_START):
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
        optimizer.zero_grad()
        train_loss=sum(train_loss)/len(train_loss)
        pbar = tqdm(enumerate(dl_val), total=len(dl_val), desc="Validating...")
        correct = 0
        total = 0
        if (config.SWA_START is not None and e>=config.SWA_START):
            final_model = swa_model
            update_bn(
                dl_train,
                final_model,
                device=config.DEVICE
            )
        else:
            final_model=model
        with torch.no_grad():
            for i, (x_batch, y_batch) in pbar:
                x_batch=x_batch.to(config.DEVICE)
                y_batch=y_batch.to(config.DEVICE)
                
                p_batch=final_model(x_batch)
                loss=loss_fn(p_batch, y_batch)
                val_loss.append(loss.item())
                pbar.set_postfix(val_loss=f"{loss.item():.4f}")
                preds=torch.argmax(p_batch, dim=-1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_loss=sum(val_loss)/len(val_loss)
        acc_now=correct/total
        if(e%config.LOG_STEP==0 and save_model_path is not None):
            model_path=f"{save_model_path}/checkpoint_{e}.pt"
            torch.save(final_model.state_dict(), model_path)
        if(best_acc<acc_now and save_model_path is not None):
            model_path=f"{save_model_path}/best_model.pt"
            torch.save(final_model.state_dict(), model_path)
            best_acc=acc_now
        print(
            f"Epoch {e}/{config.NUM_EPOCHS},",
            f"train_loss: {(train_loss):.8f} "+f"val_loss: {(val_loss):.8f} "+f"accuracy: {(acc_now):.8f} "+f"Best Acc is {best_acc}",
        )
        if(config.WANDB_TOKEN is not None):
            wandb.log({
                "train_loss_epoch": train_loss,
                "val_accuracy": acc_now,
                "val_loss":val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch":e
            }, step=global_step)
    if config.WANDB_TOKEN is not None:
        wandb.finish()
    return best_acc


def predict(model_path:str, images_path:str,save_path:str,  config=Config()):
    model=get_model(config)
    assert config.LOSS in ["CE", "ArcMargin"], "Loss is not implemented"
    model=get_model(config)
    model,loss_fn=get_loss(model,config)
    if(config.SWA_START is not None):
        model=AveragedModel(model).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    if(config.DEVICE==torch.device("cuda:0")):
        torch.cuda.empty_cache()
    ds_valid=MyDataset(images_path, mode="all", config=config)
    model.eval()
    ans={}
    for _, (img_path,  image) in enumerate(ds_valid):
        x_batch = image.unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            p_batch = model(x_batch)
            if config.LOSS =="ArcMargin":
                cosine = F.linear(
                        F.normalize(p_batch),
                        F.normalize(loss_fn.weight)
                    )
                ans[os.path.basename(img_path)] = torch.argmax(cosine, dim=-1).detach().cpu().numpy().squeeze().item()
            else:
                ans[os.path.basename(img_path)] =torch.argmax(p_batch, dim=-1).detach().cpu().numpy().squeeze().item()
    submission = pd.DataFrame(
        {"Id": ans.keys(), "Category": ans.values()}
    )
    submission.to_csv(save_path, index=False)



#------------------------------------------------------------
#MAIN FUNCTION
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Training script")
    parser.add_argument("--mode", type=str,choices=["train", "predict"], default="predict", help="Training mode or predict mode")
    parser.add_argument("--training_dir", type=str, default="bhw-1-dl-2025-2026/bhw1/trainval", help="Directory for training")
    parser.add_argument("--labels", type=str, default="bhw-1-dl-2025-2026/bhw1/labels.csv", help="path to labels csv file")
    parser.add_argument("--model_path", type=str, default="models/best_model.pt", help="Path to trained model")
    parser.add_argument("--save_model_dir", type=str, default="models", help="Path to save checkpoints")
    parser.add_argument("--pred_dir", type=str, default="bhw-1-dl-2025-2026/bhw1/test", help="Directory for testing")
    parser.add_argument("--save_submission", type=str, default="prediction.csv", help="Path to save submission")
    parser.add_argument("--wandb_token", type=str, help="wandb token")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    args=parser.parse_args()
    if args.mode=="train":
        config=Config()
        config.WANDB_TOKEN=args.wandb_token
        config.TRAININ_DIR=args.training_dir
        config.DATAPARALLEL = args.data_parallel
        if args.batch_size  is not None:
            config.BATCH_SIZE=args.batch_size 
        train_detector(args.labels, images_path=args.training_dir, config=config, save_model_path=args.save_model_dir)
    else:
        config=Config()
        config.TRAININ_DIR=str(Path(args.pred_dir).parent/"trainval")
        predict(args.model_path, args.pred_dir,save_path=args.save_submission, config=config)
