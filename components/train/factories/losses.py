import torch
import torch.nn as nn
from omegaconf import DictConfig


class MAELoss(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super(MAELoss, self).__init__()
        self.config = config
        self.igonre_index = -100

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target != self.igonre_index
        pred = pred[mask]
        target = target[mask]
        return torch.mean(torch.abs(pred - target).view(-1))
