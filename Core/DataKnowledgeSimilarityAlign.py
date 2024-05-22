import numpy as np
from cvxopt import matrix, solvers
from Core.FuzzyQuotientSpace import FuzzyRelation

"""
D: data
K: knowledge
Y: class space
"""


class DataKnowledgeSimilarityAlign(object):
    def __init__(self, FSR_D2Y: FuzzyRelation, FSR_K2Y: FuzzyRelation, alpha=0.2, margin=1e-10):
        super(DataKnowledgeSimilarityAlign, self).__init__()
        self.__FSR_D2Y = FSR_D2Y
        self.__FSR_K2Y = FSR_K2Y
        self.__alpha = alpha
        self.__margin = margin

    def GetS_DK2Y(self)-> np.ndarray:
        """

        :return:
        """
        I_arr, s_bar_arr, d_arr = self.__Sort()
        t_star = self.__QuadraticProgramming(s_bar_arr=s_bar_arr, d_arr=d_arr)
        T = self.__RecoverOptimalSolution(I_arr=I_arr, t_star=t_star)
        return T

    def __Sort(self):
        SK = self.__FSR_K2Y.mat
        n = SK.shape[0]
        ind = np.triu_indices(n=n, k=1)
        VK = np.unique(np.array(SK[ind]))  # ascending order

        max_rank = VK.shape[0]
        SK_value_rank_map = {}
        I_arr = {}
        for rank in range(max_rank):
            value = VK[rank]
            SK_value_rank_map[value] = rank
            I_arr[rank] = []

        d_arr = np.zeros(shape=[max_rank], dtype=np.float64)
        s_bar_arr = np.zeros(shape=[max_rank], dtype=np.float64)
        SD = self.__FSR_D2Y.mat
        for i in range(n):
            for j in range(i + 1, n):
                value = SK[i, j]
                rank = SK_value_rank_map[value]
                s_bar_arr[rank] += SD[i, j]
                d_arr[rank] += 1
                I_arr[rank].append([i, j])

        s_bar_arr = s_bar_arr / d_arr

        return I_arr, s_bar_arr, d_arr,

    def __QuadraticProgramming(self, s_bar_arr, d_arr):
        """
        min   (1/2)x'Px + q'x
        s.t.  Gx <= h
              Ax = b.
        :return:
        """
        n_rank = s_bar_arr.shape[0]

        P = matrix(np.diag(2 * d_arr))  # shape=[n_rank, n_rank]

        s_bar = np.expand_dims(s_bar_arr, axis=1)  # shape=[n_rank, 1]
        s_bar_D = np.matmul(np.diag(d_arr), s_bar)   # shape=[n_rank, 1]
        q = matrix(-2 * s_bar_D)  # shape=[n_rank, 1]

        G1 = - np.eye(n_rank)
        h1 = - self.__margin * np.ones(shape=[n_rank, 1])  # 0 < t_i <=> -t_i <= - margin

        G2 = np.eye(n_rank)
        h2 = self.__alpha * np.ones(shape=[n_rank, 1]) - self.__margin  # t_i < alpha  <=> t_i <= alpha-margin

        G3 = np.zeros(shape=[n_rank - 1, n_rank])
        h3 = - self.__margin * np.ones(shape=[n_rank - 1, 1])  # t_i < t_{i+1}  <=> t_i - t_{i+1} <= 0-margin

        for i in range(n_rank - 1):
            G3[i, i] = 1.0
            G3[i, i + 1] = - 1.0
        G = matrix(np.concatenate((G1, G2, G3), axis=0))
        h = matrix(np.concatenate((h1, h2, h3), axis=0))

        solvers.options['show_progress'] = False
        res = solvers.qp(P=P, q=q, G=G, h=h)
        solvers.options['show_progress'] = True
        t_star = np.array(res['x'])  # shape=[n_rank, 1]
        t_star = np.squeeze(t_star)
        return t_star

    def __RecoverOptimalSolution(self, I_arr, t_star)-> np.ndarray:
        """

        :param I_arr: dict, I_arr[0]=[[0, 1], [1, 2],...]
        :param t_star:
        :return:
        """
        n_class = self.__FSR_D2Y.mat.shape[0]
        n_rank = t_star.shape[0]
        T = np.eye(n_class)
        for rank in range(n_rank):
            pair_set = I_arr[rank]
            t = t_star[rank]
            for pair in pair_set:
                i = pair[0]
                j = pair[1]
                T[i, j] = t
                T[j, i] = t

        return T
