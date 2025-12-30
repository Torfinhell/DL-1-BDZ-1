

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import PIL.Image
from tqdm import tqdm
import torchvision.transforms.v2 as T
from torch.utils import data 
import random
import os
import typing as tp
from torch.utils import data 
import random
import argparse
from torchvision import transforms
from torchvision.models import resnet50, resnet34, resnet18, mnasnet0_5
import torch.nn.functional as F
import wandb

#PARAMETRS
class Config:
    WINDOW_SIZE=(30, 30)
    LAST_LINEAR_SIZE=200
    BATCH_SIZE=1024
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MEAN=np.array([0.56888501, 0.54452063, 0.49320272], dtype=np.float32)
    STD=np.array([0.18817873, 0.1864294 , 0.19145788], dtype=np.float32)
    ROTATE_LIMIT=45
    SCALE_LIMIT=0.1
    SHIFT_LIMIT=0.1
    LEARNING_RATE=5e-4
    ACCUM_STEP=1
    NUM_WORKERS=os.cpu_count() or 1
    LOG_STEP=5
    NUM_EPOCHS=1000
    LOSS="CE"
    MODEL="MyModel"
    NUM_CLASSES=200
    MARGIN_ARCFACE=0.20
    SCALE_ARCFACE=16
    WANDB_TOKEN=None
    WANDB_PROJECT="DL-BDZ-1_exp"
    RUN_NAME="first_run"
    OPTIMIZER="SGD"
    MOMENTUM = 0.9
    WEIGHT_DECAY=1e-4
    NUM_BLOCKS=3
    DROPOUT=0.5
    TRAININ_DIR="bhw-1-dl-2025-2026/bhw1/trainval"
    
#-----------------------------------------------------------
#MODEL
#MODEL
import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv(x)
        out = out + self.skip(x)
        out = self.relu(out)
        return self.pool(out)


class MyModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        channels = [3]
        for i in range(config.NUM_BLOCKS):
            channels.append(128 * (2 ** i))

        self.blocks = nn.ModuleList([
            ResidualConvBlock(channels[i], channels[i + 1])
            for i in range(config.NUM_BLOCKS)
        ])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], config.LAST_LINEAR_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)

    

#-----------------------------------------------------------
#LOSS

class ArcMarginLoss(nn.Module):
    def __init__(self, input_features, output_features, config:Config=Config()):
        super().__init__()
        self.weight=nn.Parameter(torch.Tensor(output_features, input_features))
        nn.init.xavier_uniform_(self.weight)
        self.m=config.MARGIN_ARCFACE
        self.s=config.SCALE_ARCFACE
    def forward(self,prev_output,labels):
        cosine=F.linear(F.normalize(prev_output), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1+1e-7, 1-1e-7))
        logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1).long(), 1)
        logits = one_hot * logits + (1 - one_hot) * cosine
        logits *= self.s
        loss = F.cross_entropy(logits, labels)
        return loss
#-----------------------------------------------------------
#TRANSFORMS
def create_transforms(config, partition: str = "train", normalise=True):
    if config.MEAN is not None and config.STD is not None:
        normalise=transforms.Normalize(config.MEAN, config.STD)
    else:
        normalise=transforms.Identity()
    if partition == "train":
        return transforms.Compose([
            transforms.Resize((60, 60)),
            transforms.RandomCrop(config.WINDOW_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),   
            transforms.RandomVerticalFlip(p=0.2),     
            # transforms.ColorJitter(   
            #     brightness=0.2,
            #     contrast=0.2,
            #     saturation=0.2,
            #     hue=0.05
            # ),
            # transforms.RandomAffine(
            #     degrees=config.ROTATE_LIMIT,
            #     translate=(config.SHIFT_LIMIT, config.SHIFT_LIMIT),
            #     scale=(1 - config.SCALE_LIMIT, 1 + config.SCALE_LIMIT),
            # ),
            transforms.ToTensor(),
            normalise,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((60, 60)),
            transforms.CenterCrop(config.WINDOW_SIZE),
            transforms.ToTensor(),
            normalise
        ])

#-----------------------------------------------------------
#Dataset
def compute_mean_std(image_paths):
    means, stds = [], []

    for path in image_paths:
        img = np.array(PIL.Image.open(path).convert("RGB")) / 255.0
        means.append(img.mean(axis=(0,1)))
        stds.append(img.std(axis=(0,1)))

    return np.mean(means, axis=0), np.mean(stds, axis=0)

class MyDataset(data.Dataset):
    
    def __init__(self, 
                 root_images:str, 
                 labels_csv:str | None=None,
                 train_fraction:float=0.8, 
                 split_seed:int=42,
                 mode:str="train",
                config:Config=Config()):
        super().__init__()
        rng=random.Random(split_seed)
        if(labels_csv is not None):
            labels={row["Id"]:row["Category"] for _, row in pd.read_csv(labels_csv).iterrows()}
            self.paths=[]
            self.labels=[]
            for category in tqdm(range(config.NUM_CLASSES), desc="Loading classes paths"):
                paths=sorted([id for id, cat in labels.items() if cat==category])
                split_train=int(train_fraction*len(paths))
                if(mode=="train"):
                    paths=paths[:split_train]
                elif(mode=="valid"):
                    paths=paths[split_train:]
                elif(mode!="all"):
                    raise ValueError("Mode is not train or valid or all")
                self.paths.extend(paths)
                self.labels.extend([category]*len(paths))
            combined = list(zip(self.paths, self.labels))
            rng.shuffle(combined)
            self.paths, self.labels = zip(*combined)
        else:
            self.labels=None
            self.paths=sorted([file for file  in os.listdir(root_images) if file.endswith(".jpg")])
        self.paths=[f"{root_images}/{file}" for file in self.paths]
        if config.MEAN is None or config.STD is None:
            image_paths = [
                os.path.join(config.TRAININ_DIR, f)
                for f in os.listdir(config.TRAININ_DIR)
                if f.endswith(".jpg")
            ]
            config.MEAN, config.STD = compute_mean_std(image_paths)
        self._transform=create_transforms(config, mode) 
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index:int):
        """
        Takes image (h, w, 3), points array (14, 2)
        Returns tensor image (3,100, 100), and tensor array(14, 2) and label when transform  and file-poinits is specified
        Returns img_path and numpy.image(3, 100, 100) when labels_csv is not specified 
        """
        img_path=self.paths[index]
        image = PIL.Image.open(img_path).convert("RGB")
        if(self._transform is not None):
            image=self._transform(image)
        if(self.labels is not None):
            label=self.labels[index]
            return image, label
        return (img_path,image)


#-----------------------------------------------------------





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
    if(config.MODEL=="MyModel"):
        model=MyModel(config)
    elif(config.MODEL=="RESNET50"):
        model=resnet50(weights=None)
        model.fc=nn.Linear(model.fc.in_features,  config.NUM_CLASSES)
    elif(config.MODEL=="RESNET34"):
        model=resnet34(weights=None)
        model.fc=nn.Linear(model.fc.in_features,  config.NUM_CLASSES)
    elif(config.MODEL=="RESNET18"):
        model=resnet18(weights=None)
        model.fc=nn.Linear(model.fc.in_features,  config.NUM_CLASSES)
    elif config.MODEL == "MNASNET0_5":
        model = mnasnet0_5(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            config.NUM_CLASSES
        )
    assert config.LOSS in ["CE", "ArcMargin"], "Loss is not implemented"
    if config.LOSS=="CE":
        loss_fn=torch.nn.CrossEntropyLoss().to(config.DEVICE)#TODO Another Loss
    elif config.LOSS =="ArcMargin":
        if config.MODEL=="MyModel":
            loss_fn = ArcMarginLoss(config.LAST_LINEAR_SIZE, config.NUM_CLASSES, config).to(config.DEVICE)
        elif config.MODEL in ["RESNET50","RESNET34","RESNET18"]:
            loss_fn = ArcMarginLoss(config.LAST_LINEAR_SIZE, config.NUM_CLASSES, config).to(config.DEVICE)
            model.fc = nn.Linear(model.fc.in_features, config.LAST_LINEAR_SIZE)
        elif config.MODEL=="MNASNET0_5":
            loss_fn = ArcMarginLoss(config.LAST_LINEAR_SIZE, config.NUM_CLASSES, config).to(config.DEVICE)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.LAST_LINEAR_SIZE)
    model = model.to(config.DEVICE)
    if(config.OPTIMIZER=="AdamW"):
        optimizer=torch.optim.AdamW(model.parameters() if config.LOSS !="ArcMargin" else list(model.parameters()) + list(loss_fn.parameters()), lr=config.LEARNING_RATE)
    elif(config.OPTIMIZER=="SGD"):
        optimizer=torch.optim.SGD(model.parameters() if config.LOSS !="ArcMargin" else list(model.parameters()) + list(loss_fn.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, nesterov=True, momentum=config.MOMENTUM)
    else:
        raise NotImplementedError("Optimizer is not implemented")
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    best_acc=0
    ds_val=MyDataset(images_path, labels_csv,mode="valid",  config=config)
    dl_val=data.DataLoader(
        ds_val, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=config.NUM_WORKERS,
    )
    for e in range(config.NUM_EPOCHS):
        model.train()
        train_loss=[]
        val_loss=[]
        accuracy=[]
        optimizer.zero_grad()
        pbar = tqdm(enumerate(dl_train), total=len(dl_train), desc="Training...")
        for i, (x_batch, y_batch) in pbar:
            x_batch=x_batch.to(config.DEVICE)
            y_batch=y_batch.to(config.DEVICE)
            p_batch=model(x_batch)
            loss=loss_fn(p_batch, y_batch)
            loss = loss / config.ACCUM_STEP
            train_loss.append(loss.item())
            loss.backward()
            if((i+1)%config.ACCUM_STEP==0):
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_postfix(train_loss=f"{loss.item():.4f}")
        optimizer.zero_grad()
        scheduler.step()
        train_loss=sum(train_loss)/len(train_loss)
        pbar = tqdm(enumerate(dl_val), total=len(dl_val), desc="Validating...")
        with torch.no_grad():
            for i, (x_batch, y_batch) in pbar:
                x_batch=x_batch.to(config.DEVICE)
                y_batch=y_batch.to(config.DEVICE)
                p_batch=model(x_batch)
                loss=loss_fn(p_batch, y_batch)
                val_loss.append(loss.item())
                pbar.set_postfix(val_loss=f"{loss.item():.4f}")
                if config.LOSS !="ArcMargin":
                    accuracy.append((torch.argmax(p_batch, dim=-1) == y_batch).sum().item())
                else:
                    cosine = F.linear(
                        F.normalize(p_batch),
                        F.normalize(loss_fn.weight)
                    )
                    preds = cosine.argmax(dim=1)
                    accuracy.append((preds == y_batch).sum().item())
        val_loss=sum(val_loss)/len(val_loss)
        acc_now=sum(accuracy)/len(ds_val)
        if(e%config.LOG_STEP==0 and save_model_path is not None):
            model_path=f"{save_model_path}/checkpoint_{e}.pt"
            if(config.LOSS!="ArcMargin"):
                torch.save(model.state_dict(), model_path)
            else:
                torch.save({
                    "model":model.state_dict(), 
                    "arcface_weight":loss_fn.weight.data
                }, model_path)
        if(best_acc<acc_now):
            if(save_model_path is not None):
                model_path=f"{save_model_path}/best_model.pt"
                if(config.LOSS!="ArcMargin"):
                    torch.save(model.state_dict(), model_path)
                else:
                    torch.save({
                        "model":model.state_dict(), 
                        "arcface_weight":loss_fn.weight.data
                    }, model_path)

            best_acc=acc_now
        print(
            f"Epoch {e}/{config.NUM_EPOCHS},",
            f"train_loss: {(train_loss):.8f} "+f"val_loss: {(val_loss):.8f} "+f"accuracy: {(acc_now):.8f} "+f"Best Acc is {best_acc}",
        )
        if(config.WANDB_TOKEN is not None):
            if config.WANDB_TOKEN is not None:
                log_dict = {
                    "train_loss": train_loss,
                    "accuracy": acc_now,
                    "val_loss":val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                wandb.log(log_dict)
    if config.WANDB_TOKEN is not None:
        wandb.finish()
    return best_acc


def predict(model_path:str, images_path:str,save_path:str,  config=Config()):
    if(config.MODEL=="MyModel"):
        model=MyModel(config)
    elif(config.MODEL=="RESNET50"):
        model=resnet50(weights=None)
        model.fc=nn.Linear(model.fc.in_features,  config.NUM_CLASSES)
    elif(config.MODEL=="RESNET34"):
        model=resnet34(weights=None)
        model.fc=nn.Linear(model.fc.in_features,  config.NUM_CLASSES)
    elif(config.MODEL=="RESNET18"):
        model=resnet18(weights=None)
        model.fc=nn.Linear(model.fc.in_features,  config.NUM_CLASSES)
    elif config.MODEL == "MNASNET0_5":
        model = mnasnet0_5(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            config.NUM_CLASSES
        )
    assert config.LOSS in ["CE", "ArcMargin"], "Loss is not implemented"
    if config.LOSS=="CE":
        loss_fn=torch.nn.CrossEntropyLoss().to(config.DEVICE)#TODO Another Loss
    elif config.LOSS =="ArcMargin":
        if config.MODEL=="MyModel":
            loss_fn = ArcMarginLoss(config.LAST_LINEAR_SIZE, config.NUM_CLASSES, config).to(config.DEVICE)
        elif config.MODEL == "MNASNET0_5":
            loss_fn=ArcMarginLoss(config.LAST_LINEAR_SIZE, config.NUM_CLASSES, config).to(config.DEVICE)
            model.classifier[1]=nn.Linear(model.classifier[1].in_features,  config.LAST_LINEAR_SIZE)
        else:
            loss_fn=ArcMarginLoss(config.LAST_LINEAR_SIZE, config.NUM_CLASSES, config).to(config.DEVICE)
            model.fc=nn.Linear(model.fc.in_features,  config.LAST_LINEAR_SIZE)
    model = model.to(config.DEVICE)
    if(config.LOSS!="ArcMargin"):
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    else:
        ckpt = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model"])
        loss_fn.weight.data.copy_(ckpt["arcface_weight"])
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
    parser.add_argument("--wandb_token", type=str, help="Path to save submission")
    args=parser.parse_args()
    if args.mode=="train":
        config=Config()
        config.WANDB_TOKEN=args.wandb_token
        train_detector(args.labels, images_path=args.training_dir, config=config, save_model_path=args.save_model_dir)
    else:
        config=Config()
        config.MODEL="RESNET18"
        config.LOSS = "ArcMargin"
        predict(args.model_path, args.pred_dir,save_path=args.save_submission, config=config)
