import torch
import torch.nn as nn


class ListNetLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temp = temperature

    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred = pred[mask > 0]
            target = target[mask > 0]
        if len(pred) < 5:
            return torch.tensor(0.0, device=pred.device)
        return -torch.sum(torch.softmax(target / self.temp, dim=0) * torch.log_softmax(pred / self.temp, dim=0))


class V4CombinedLoss(nn.Module):
    def __init__(self, listnet_weight=0.7, ic_weight=0.3, temperature=1.0):
        super().__init__()
        self.lw = listnet_weight
        self.iw = ic_weight
        self.listnet = ListNetLoss(temperature)

    def forward(self, pred, target, mask=None):
        if mask is not None:
            valid = mask > 0
            pred_valid = pred[valid]
            target_valid = target[valid]
        else:
            pred_valid = pred
            target_valid = target

        if len(pred_valid) < 5:
            return torch.tensor(0.0, device=pred.device), 0.0, 0.0

        pred_mean = pred_valid - pred_valid.mean()
        target_mean = target_valid - target_valid.mean()
        ic = (pred_mean * target_mean).mean() / ((pred_mean.std() + 1e-8) * (target_mean.std() + 1e-8))
        ic_loss = 1 - ic

        if self.lw > 0:
            listnet_loss = self.listnet(pred, target, mask)
            total = self.lw * listnet_loss + self.iw * ic_loss
            return total, listnet_loss.item(), ic.item()

        return self.iw * ic_loss, 0.0, ic.item()
