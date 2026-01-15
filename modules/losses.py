from torch import nn
import torch.nn.functional as F
import torch
from .config import Config
class ArcModel(nn.Module):
    def __init__(self, model, config:Config):
        super().__init__()
        self.backbone=model        
        if config.MODEL in ["RESNET50","RESNET34","RESNET18"]:
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, config.LAST_LINEAR_SIZE)
        elif config.MODEL=="MNASNET0_5":
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, config.LAST_LINEAR_SIZE)
        if(config.LOSS=="ArcMargin"):
            self.arc_model=ArcMargin(config.LAST_LINEAR_SIZE, config.NUM_CLASSES)
        else:
            raise NotImplementedError("Arc Model is not implemented")
    def forward(self,x):
        return self.arc_model(self.backbone(x))

class ArcMargin(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self,prev_output):
        cosine=F.linear(F.normalize(prev_output, dim=1), F.normalize(self.weight, dim=1))
        return cosine
class ArcmarginLoss(nn.Module):
    def __init__(self, config:Config=Config()):
        super().__init__()
        self.m=config.MARGIN_ARCFACE
        self.s=config.SCALE_ARCFACE
    def forward(self, cosine, labels):
        sine = torch.sqrt(1.0 - cosine**2)
        margin_logits = cosine * torch.cos(torch.tensor(self.m)) - sine * torch.sin(torch.tensor(self.m)) 
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1).long(), 1)
        logits = one_hot * margin_logits + (1 - one_hot) * cosine
        logits *= self.s
        loss = F.cross_entropy(logits, labels)
        return loss
def get_loss(model,config:Config):
    assert config.LOSS in ["CE", "ArcMargin"], "Loss is not implemented"
    if config.LOSS=="CE":
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.1)#TODO Another Loss
    elif config.LOSS =="ArcMargin":
        model=ArcModel(model, config)
        loss_fn=ArcmarginLoss()
    return model, loss_fn
    