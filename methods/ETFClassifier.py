# /home/minkyoon/2024_FeTrIL_Fl/MFCL/MFCL_Fetril/models/ETFClassifier.py
import math
import torch
import torch.nn as nn
import numpy as np
from models.DRLoss import DRLoss

def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec

class ETFClassifier(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(ETFClassifier, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        orth_vec = generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1)))
        self.etf_vec = etf_vec.to('cuda')
        #self.etf_vec('etf_vec', etf_vec)
        self.drloss=DRLoss()
        etf_rect = torch.ones((1, self.num_classes), dtype=torch.float32)
        self.etf_rect = etf_rect

    
    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x
    
    def get_drloss(self, x, labels, with_len=False):
        x=self.pre_logits(x)
        if with_len:
            etf_vec = self.etf_vec * self.etf_rect.to(device=self.etf_vec.device)
            target = (etf_vec * self.produce_training_rect(labels, self.num_classes))[:, labels].t()
        else :
            target = self.etf_vec[:, labels].t()
        loss = self.drloss(x, target)
        return loss
    
    def forward(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        cls_score = x @ self.etf_vec
        
        return cls_score

    def produce_training_rect(self, label: torch.Tensor, num_classes: int):
    # rank, world_size = get_dist_info()
    # if world_size > 0:
    #     recv_list = [None for _ in range(world_size)]
    #     dist.all_gather_object(recv_list, label.cpu())
    #     new_label = torch.cat(recv_list).to(device=label.device)
    #     label = new_label
        uni_label, count = torch.unique(label, return_counts=True)
        batch_size = label.size(0)
        uni_label_num = uni_label.size(0)
        assert batch_size == torch.sum(count)
        gamma = torch.tensor(batch_size / uni_label_num, device=label.device, dtype=torch.float32)
        rect = torch.ones(1, num_classes).to(device=label.device, dtype=torch.float32)
        rect[0, uni_label] = torch.sqrt(gamma / count)
        return rect
