# /home/minkyoon/2024_FeTrIL_Fl/MFCL/MFCL_Fetril/models/DRLoss.py
import torch
import torch.nn as nn

class DRLoss(nn.Module):
    def __init__(self, loss_weight=10):
        super(DRLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, feat, target):
        dot = torch.sum(feat * target, dim=1)
        h_norm2 = torch.ones_like(dot)
        m_norm2 = torch.ones_like(dot)
        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)
        return loss * self.loss_weight