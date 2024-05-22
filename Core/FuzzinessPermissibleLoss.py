import torch
from torch import nn, Tensor

"""
Cla: Class
Equ: Equivalent
Rel: Relation
"""


class FuzzinessPermissibleLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8):
        super(FuzzinessPermissibleLoss, self).__init__()
        self.__alpha = alpha
        self.__beta = beta

    def ComputeLoss(self, FSR_Mat: Tensor, labels: Tensor):
        """

        :param FSR_Mat:
        :param labels:
        :return:
        """
        max_ind = torch.max(labels) + 1
        I_mat = torch.eye(n=max_ind, m=max_ind, device=FSR_Mat.device) == 1
        is_same_class = I_mat[labels, :]
        is_same_class = is_same_class[:, labels]
        loss_eq = torch.sum(torch.relu(self.__beta - FSR_Mat[is_same_class]))
        loss_neq = torch.sum(torch.relu(FSR_Mat[~is_same_class] - self.__alpha))
        return (loss_eq + loss_neq) / FSR_Mat.numel()

    def __repr__(self):
        return 'FuzzinessPermissibleLoss(__alpha=' + str(self.__alpha) + ', __beta=' + str(self.__beta) + ')'


