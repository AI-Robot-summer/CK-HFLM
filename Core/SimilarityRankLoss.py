import torch
from torch import nn, Tensor

"""
SR: Similarity Relation
ER: Equivalent Relation
"""


class SimilarityRankLoss(nn.Module):
    def __init__(self, margin=1e-5):
        super(SimilarityRankLoss, self).__init__()
        self.__margin = margin
        self.__class_sim_mat = None

    def SetClassSimMat(self, class_sim_mat: Tensor):
        """
        Use only the sorting of values, the specific value does not matter.
        :param class_sim_mat: shape=[num_class, num_class]
        :return:
        """
        self.__class_sim_mat = class_sim_mat

    def ComputeLoss(self, FSR_Mat: Tensor, labels: Tensor):
        """

        :param FSR_Mat: shape=[n_sample, n_sample]
        :param labels: shape=[n_sample], labels[i] in {0, 1, 2,..., num_class-1}
        :return:
        """

        n_sample = FSR_Mat.shape[0]
        ind = torch.arange(start=0, end=n_sample, device=FSR_Mat.device)
        I4 = torch.meshgrid([ind, ind, ind, ind])
        n4 = I4[0].numel()
        Ii = I4[0].reshape(shape=[n4])
        Ij = I4[1].reshape(shape=[n4])
        Ik = I4[2].reshape(shape=[n4])
        Il = I4[3].reshape(shape=[n4])
        S_xi_xj = FSR_Mat[Ii, Ij]
        S_xk_xl = FSR_Mat[Ik, Il]
        S_kl_ij = S_xk_xl - S_xi_xj

        S_yi_yj = self.__class_sim_mat[labels[Ii], labels[Ij]]
        S_yk_yl = self.__class_sim_mat[labels[Ik], labels[Il]]
        sign_ij_kl = torch.sign(S_yi_yj - S_yk_yl)

        I_neq = sign_ij_kl != 0
        loss_eq = torch.sum(input=torch.abs(S_kl_ij[~I_neq]))

        loss_neq = torch.relu(input=sign_ij_kl[I_neq] * S_kl_ij[I_neq] + self.__margin)
        loss_neq = torch.sum(input=loss_neq)

        loss = (loss_eq + loss_neq) / n4
        return loss

    def __repr__(self):
        return 'SimilarityRankLoss(__margin=' + str(self.__margin) + ')'
