import numpy as np
import pandas as pd
import PIL.Image
from tqdm import tqdm
from torch.utils import data 
import random
import os
from torch.utils import data 
import random
from torchvision import transforms
from .config import Config
from copy import deepcopy
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
            transforms.RandAugment(num_ops=config.NUM_OPS_AUGS, magnitude=config.MAGNITUDE, num_magnitude_bins=config.MAGNITUDE+1),
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
        self.config=deepcopy(config)
        self.mode=mode
        if(labels_csv is not None):
            labels={row["Id"]:row["Category"] for _, row in pd.read_csv(labels_csv).iterrows()}
            self.paths=[]
            self.labels=[]
            for category in tqdm(range(self.config.NUM_CLASSES), desc="Loading classes paths"):
                paths=sorted([id for id, cat in labels.items() if cat==category])
                rng.shuffle(paths)
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
        if self.config.MEAN is None or self.config.STD is None:
            image_paths = [
                os.path.join(self.config.TRAININ_DIR, f)
                for f in os.listdir(self.config.TRAININ_DIR)
                if f.endswith(".jpg")
            ]
            self.config.MEAN, self.config.STD = compute_mean_std(image_paths)
        self._transform=create_transforms(self.config, mode) 
        if(config.SCHEDULER=="OneCycle"):
            self.update_transform(0)
        
    def update_transform(self,new_magnitude):
        self.config.MAGNITUDE=new_magnitude
        self._transform=create_transforms(self.config, self.mode) 

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
