import torch
from torch import nn, Tensor


class IntervalRegressionLoss(nn.Module):
    def __init__(self):
        super(IntervalRegressionLoss, self).__init__()
        self.__class_smi_UB = None
        self.__class_smi_LB = None

    def SetClassSim_LUB(self, class_smi_UB: Tensor, class_smi_LB: Tensor):
        """

        :param class_smi_UB: shape=[num_class, num_class]
        :param class_smi_LB: shape=[num_class, num_class]
        :return:
        """
        self.__class_smi_UB = class_smi_UB
        self.__class_smi_LB = class_smi_LB

    def ComputeLoss(self, FSR_Mat: Tensor, labels: Tensor):
        """

        :param FSR_Mat: shape=[batch_size, batch_size]
        :param labels: shape=[batch_size], labels[i] in {0, 1, 2, ..., num_class -1}
        :return:
        """

        UB_Mat = self.__class_smi_UB[labels, :]
        UB_Mat = UB_Mat[:, labels]

        LB_Mat = self.__class_smi_LB[labels, :]
        LB_Mat = LB_Mat[:, labels]

        ind = FSR_Mat < LB_Mat
        loss1 = torch.sum(LB_Mat[ind] - FSR_Mat[ind])

        ind = FSR_Mat > UB_Mat
        loss2 = torch.sum(FSR_Mat[ind] - UB_Mat[ind])

        loss = (loss1 + loss2) / FSR_Mat.numel()
        return loss
