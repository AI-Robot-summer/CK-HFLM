import torch
from torch import nn


class RegularizationTerm(nn.Module):
    def __init__(self):
        super(RegularizationTerm, self).__init__()

    def forward(self, m: nn.Module):
        device = None
        for para in m.parameters():
            device = para.device
            break
        para_num = torch.tensor(data=0.0, device=device)
        reg = torch.tensor(data=0.0, device=device, requires_grad=True)
        for para in m.parameters():
            reg = reg + torch.sum(input=para * para)
            para_num += para.numel()

        return 0.5 * reg / para_num
