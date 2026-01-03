from torch import nn
import torch.nn.functional as F
import torch
from .config import Config
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
def get_loss(config:Config, model=None):
    assert config.LOSS in ["CE", "ArcMargin"], "Loss is not implemented"
    if config.LOSS=="CE":
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(config.DEVICE)#TODO Another Loss
    elif config.LOSS =="ArcMargin":
        if config.MODEL=="MyModel":
            loss_fn = ArcMarginLoss(config.LAST_LINEAR_SIZE, config.NUM_CLASSES, config).to(config.DEVICE)
        elif config.MODEL in ["RESNET50","RESNET34","RESNET18"]:
            loss_fn = ArcMarginLoss(config.LAST_LINEAR_SIZE, config.NUM_CLASSES, config).to(config.DEVICE)
            model.fc = nn.Linear(model.fc.in_features, config.LAST_LINEAR_SIZE)
        elif config.MODEL=="MNASNET0_5":
            loss_fn = ArcMarginLoss(config.LAST_LINEAR_SIZE, config.NUM_CLASSES, config).to(config.DEVICE)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.LAST_LINEAR_SIZE)
    return loss_fn, model
    