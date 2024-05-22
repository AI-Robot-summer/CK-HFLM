import torch
from torch import nn, Tensor


class FuzzySimilarityRelationNetwork(nn.Module):
    """
    Gaussian Kernel
    """

    def __init__(self):
        """
        """
        super(FuzzySimilarityRelationNetwork, self).__init__()

    def ComputeFSR_Mat(self, H: Tensor) -> Tensor:
        """

        :param H: shape=[num_sam, num_fea]
        :return S: shape=[num_sam, num_sam], 0 <= S[ij] <= 1, S[ii]=1
        """
        pass

    def ComputeFSR_Mat2(self, H1: Tensor, H2: Tensor) -> Tensor:
        """

        :param H1: shape=[n1_sam, num_fea]
        :param H2: shape=[n2_sam, num_fea]
        :return S: shape=[n1_sam, n2_sam], 0 <= S[ij] <= 1
        """
        pass

    def ComputeFSR_Mat2_diag(self, H1: Tensor, H2: Tensor) -> Tensor:
        """

        :param H1: shape=[n_sam, num_fea]
        :param H2: shape=[n_sam, num_fea]
        :return s: shape=[n_sam]
        """
        pass


class GaussianKernel(FuzzySimilarityRelationNetwork):
    """
    Gaussian Kernel: S[ij]=exp(-delta * ||x_i-x_j||_2^2)
    """

    def __init__(self, kernel_para=1.0):
        """

        :param kernel_para: > 0
        """
        super(GaussianKernel, self).__init__()
        self.__kernel_para = kernel_para

    def ComputeFSR_Mat(self, H: Tensor) -> Tensor:
        """

        :param H: shape=[num_sam, num_fea]
        :return S: shape=[num_sam, num_sam], 0 <= S[ij] <= 1, S[ii]=1
        """
        return self.ComputeFSR_Mat2(H1=H, H2=H)

    def ComputeFSR_Mat2(self, H1: Tensor, H2: Tensor) -> Tensor:
        """

        :param H1: shape=[n1_sam, num_fea]
        :param H2: shape=[n2_sam, num_fea]
        :return S: shape=[n1_sam, n2_sam], 0 <= S[ij] <= 1
        """
        H11 = torch.sum(input=H1 * H1, dim=1).view(size=[H1.shape[0], 1])  # shape=[n1_sam, 1]
        H22 = torch.sum(input=H2 * H2, dim=1).view(size=[1, H2.shape[0]])  # shape=[1, n2_sam]
        H12 = torch.matmul(input=H1, other=H2.T)  # shape=[batch1_size, batch2_size]
        D_2 = H11 + H22 - 2 * H12
        return torch.exp(input=-self.__kernel_para * D_2)


class SoftmaxCos(FuzzySimilarityRelationNetwork):
    """
    softmax-cos: S[ij]=(h_i^Th_j) / (||h_i||_2 * ||h_j||_2),
                 h_j=softmax(x_i), h_j=softmax(x_j)
    """

    def __init__(self, epsilon=1e-15):
        """

        :param epsilon: threshold,  Avoid numerical overflow
        """
        super(SoftmaxCos, self).__init__()
        self.__cos_sim = CosSimilarity(epsilon=epsilon)

    def ComputeFSR_Mat(self, H: Tensor) -> Tensor:
        """

        :param H: shape=[num_sam, num_fea]
        :return S: shape=[num_sam, num_sam], 0 <= S[ij] <= 1, S[ii]=1
        """
        H = torch.softmax(input=H, dim=1)
        return self.__cos_sim.SquareMatrix(H=H)

    def ComputeFSR_Mat2(self, H1: Tensor, H2: Tensor) -> Tensor:
        """

        :param H1: shape=[n1_sam, num_fea]
        :param H2: shape=[n2_sam, num_fea]
        :return S: shape=[n1_sam, n2_sam], 0 <= S[ij] <= 1
        """
        H1 = torch.softmax(input=H1, dim=1)
        H2 = torch.softmax(input=H2, dim=1)
        return self.__cos_sim.Matrix(H1=H1, H2=H2)

    def ComputeFSR_Mat2_diag(self, H1: Tensor, H2: Tensor) -> Tensor:
        """

        :param H1: shape=[n_sam, num_fea]
        :param H2: shape=[n_sam, num_fea]
        :return s: shape=[n_sam], 0 <= S[ij] <= 1
        """
        H1 = torch.softmax(input=H1, dim=1)
        H2 = torch.softmax(input=H2, dim=1)
        return self.__cos_sim.Matrix_diag(H1=H1, H2=H2)


class SigmoidCos(FuzzySimilarityRelationNetwork):
    """
    sigmoid-cos: S[ij]=(h_i^Th_j) / (||h_i||_2 * ||h_j||_2),
                 h_j=sigmoid(x_i), h_j=sigmoid(x_j)
    """

    def __init__(self, epsilon=1e-15):
        """

        :param epsilon: threshold,  Avoid numerical overflow
        """
        super(SigmoidCos, self).__init__()
        self.__cos_sim = CosSimilarity(epsilon=epsilon)

    def ComputeFSR_Mat(self, H: Tensor) -> Tensor:
        """

        :param H: shape=[num_sam, num_fea]
        :return S: shape=[num_sam, num_sam], 0 <= S[ij] <= 1, S[ii]=1
        """
        H = torch.sigmoid(input=H)
        return self.__cos_sim.SquareMatrix(H=H)

    def ComputeFSR_Mat2(self, H1: Tensor, H2: Tensor) -> Tensor:
        """

        :param H1: shape=[n1_sam, num_fea]
        :param H2: shape=[n2_sam, num_fea]
        :return S: shape=[n1_sam, n2_sam], 0 <= S[ij] <= 1
        """
        H1 = torch.sigmoid(input=H1)
        H2 = torch.sigmoid(input=H2)
        return self.__cos_sim.Matrix(H1=H1, H2=H2)

    def ComputeFSR_Mat2_diag(self, H1: Tensor, H2: Tensor) -> Tensor:
        """

        :param H1: shape=[n_sam, num_fea]
        :param H2: shape=[n_sam, num_fea]
        :return s: shape=[n_sam]
        """
        H1 = torch.sigmoid(input=H1)
        H2 = torch.sigmoid(input=H2)
        return self.__cos_sim.Matrix_diag(H1=H1, H2=H2)


class CosSimilarity(nn.Module):
    def __init__(self, epsilon=1e-15):
        super(CosSimilarity, self).__init__()
        self.__epsilon = epsilon

    def SquareMatrix(self, H: Tensor) -> Tensor:
        """

        :param H: shape=[n_sam, n_fea]
        :return S: shape=[n_sam, n_sam]
        """

        S = torch.matmul(input=H, other=H.T)  # shape=[n_sam, n_sam]

        d = torch.diag(input=S)  # shape=[n_sam]
        d = torch.sqrt(input=d)
        d = torch.clip(input=d, min=self.__epsilon)  # Avoid numerical overflow
        d = 1.0 / d
        D = torch.diag_embed(input=d)  # shape=[n_sam, n_sam]

        S = torch.mm(input=D, mat2=S)
        S = torch.mm(input=S, mat2=D)
        S = torch.clip(input=S, min=0.0, max=1.0)
        return S

    def Matrix(self, H1: Tensor, H2: Tensor) -> Tensor:
        """

        :param H1: shape=[n1_sam, n_fea]
        :param H2: shape=[n2_sam, n_fea]
        :return S: shape=[n1_sam, n2_sam]
        """
        d1 = torch.sum(input=H1 * H1, dim=1)  # shape=[n1_sam]
        d1 = torch.sqrt(input=d1)
        d1 = torch.clip(input=d1, min=self.__epsilon)  # Avoid numerical overflow
        d1 = 1.0 / d1
        D1 = torch.diag_embed(input=d1)  # shape=[n1_sam, n1_sam]

        d2 = torch.sum(input=H2 * H2, dim=1)  # shape=[n2_sam]
        d2 = torch.sqrt(input=d2)
        d2 = torch.clip(input=d2, min=self.__epsilon)  # Avoid numerical overflow
        d2 = 1.0 / d2
        D2 = torch.diag_embed(input=d2)  # shape=[n2_sam, n2_sam]

        S = torch.matmul(input=H1, other=H2.T)  # shape=[n1_sam, n2_sam]
        S = torch.mm(input=D1, mat2=S)
        S = torch.mm(input=S, mat2=D2)
        S = torch.clip(input=S, min=0.0, max=1.0)
        return S

    def Matrix_diag(self, H1: Tensor, H2: Tensor) -> Tensor:
        """

        :param H1: shape=[n_sam, n_fea]
        :param H2: shape=[n_sam, n_fea]
        :return s: shape=[n_sam]
        """
        d1 = torch.sum(input=H1 * H1, dim=1)  # shape=[n_sam]
        d1 = torch.sqrt(input=d1)
        d1 = torch.clip(input=d1, min=self.__epsilon)  # Avoid numerical overflow

        d2 = torch.sum(input=H2 * H2, dim=1)  # shape=[n_sam]
        d2 = torch.sqrt(input=d2)
        d2 = torch.clip(input=d2, min=self.__epsilon)  # Avoid numerical overflow

        d12 = d1 * d2
        s = torch.sum(input=H1 * H2, dim=1)  # shape= [n_sam]
        s = s / d12
        s = torch.clip(input=s, min=0.0, max=1.0)
        return s

    def cos_sim(self, x: Tensor, y: Tensor) -> Tensor:
        """

        :param x: [n_fae]
        :param y: [n_fea]
        :return:
        """
        fz = torch.sum(x * y)

        fm1 = torch.sqrt(input=torch.sum(x*x))
        fm1 = torch.clip(input=fm1, min=self.__epsilon)

        fm2 = torch.sqrt(input=torch.sum(y*y))
        fm2 = torch.clip(input=fm2, min=self.__epsilon)

        fm = fm1 * fm2

        s = fz / fm
        s = torch.clip(input=s, min=0.0, max=1.0)
        return s
