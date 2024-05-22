import numpy as np
from nltk.corpus import wordnet as wn

from Core.DataSet import ClassSpaceInformation
from Core.Kit import Tool


class ClassSimilarity(object):
    def __init__(self):
        super(ClassSimilarity, self).__init__()

    @staticmethod
    def __WordNetSimilarity(synset1, synset2, sim_type: str, infor_content=None):
        # print(str(synset1) + ' <--> ' + str(synset2))
        if sim_type == 'path_similarity':
            return wn.path_similarity(synset1, synset2)
        elif sim_type == 'lch_similarity':
            return wn.lch_similarity(synset1, synset2)
        elif sim_type == 'wup_similarity':
            return wn.wup_similarity(synset1, synset2)
        else:
            raise Exception('sim_type == ' + sim_type + 'is error.')

    @staticmethod
    def WordNetSimilarity(class_inf: ClassSpaceInformation, sim_type: str):
        """

        :param class_inf:
        :param sim_type: {'path_similarity',  'lch_similarity', 'wup_similarity'}
        :return:
        """

        class_space = np.unique(np.array(class_inf.GetClassID_list()))
        class_num = class_inf.GetClassNum()
        mat = -np.inf * np.ones(shape=[class_num, class_num])
        for i in range(class_num):
            c1 = class_space[i]
            word = class_inf.class_word_list[c1]
            pos = class_inf.class_pos_list[c1]
            nn = class_inf.class_nn_list[c1]
            synset_c1 = wn.synset(word + '.' + pos + '.' + nn)
            mat[c1, c1] = ClassSimilarity.__WordNetSimilarity(synset1=synset_c1, synset2=synset_c1, sim_type=sim_type)
            for j in range(i+1, class_num):
                c2 = class_space[j]
                word = class_inf.class_word_list[c2]
                pos = class_inf.class_pos_list[c2]
                nn = class_inf.class_nn_list[c2]
                synset_c2 = wn.synset(word + '.' + pos + '.' + nn)
                s = ClassSimilarity.__WordNetSimilarity(synset1=synset_c1, synset2=synset_c2, sim_type=sim_type)
                mat[c1, c2] = s
                mat[c2, c1] = s
        return mat

    @staticmethod
    def ComputeSimilarity(A: np.array, sim_type: str, sigma=1.0):
        """

        :param A: shape=[n_classes, n_attributes]
        :param sim_type: {'Cos', 'Gaussian-kernel'}
        :param sigma: parameter of gaussian kernel
        :return:
        """
        n_class = A.shape[0]

        # 1. Similarity matrix
        S = None
        if sim_type == 'Cos':
            # 1.1 Cos
            d = np.sum(A * A, axis=1)
            d = np.sqrt(d)
            d = 1.0 / d
            D = np.diag(d)
            S = np.matmul(A, np.transpose(A))
            S = np.matmul(np.matmul(D, S), D)  # -1 <= S[i, j] <= 1, S[i, i] = 1.0
            # 2. [-1, 1] --> [0, 1]
            S = (S + 1.0) / 2.0  # 0 <= S[i, j] <= 1, S[i, i] = 1.0
        elif sim_type == 'Gaussian-kernel':
            # 1.2 2-norm distance ^2
            D = np.zeros(shape=[n_class, n_class])
            for i in range(n_class):
                for j in range(i + 1, n_class):
                    d = A[i, :] - A[j, :]
                    d = np.sum(d * d)
                    D[i, j] = d
                    D[j, i] = d
            # Gaussian kernel
            sigma_ = 2 * sigma * sigma
            S = np.exp(-D / sigma_)
        else:
            raise Exception('sim_type ==' + str(sim_type) + '.')
        return S

    @staticmethod
    def KNN_Graph(S: np.array, k: int, knn_type='or'):
        """

        :param S: shape=[n_class, n_class]
        :param k:
        :param knn_type: {'or', 'and'}
        :return:
        """
        n_class = S.shape[0]
        if k >= n_class:
            n = n_class - 1
        is_knn = np.zeros(shape=[n_class, n_class])
        for i in range(n_class):
            si = S[i, :]
            ind = np.argsort(-si)
            is_knn[i, ind[0: k + 1]] = 1

        if knn_type == 'or':
            is_knn = is_knn + np.transpose(is_knn) > 0.0  # or knn
        elif knn_type == 'and':
            is_knn = is_knn + np.transpose(is_knn) == 2.0  # or knn
        else:
            raise Exception('knn_type == ' + str(knn_type) + '.')

        is_knn = is_knn + 0.0
        S = S * is_knn
        return S

    @staticmethod
    def ClassGraph2LU_Bound(S: np.array, alpha=0.2, beta=0.8, min_sim=1e-3):
        """

        :param S: shape=[n_class, n_class], S[i, i] = 1
        :param alpha:
        :param beta:
        :param min_sim:
        :return:
        """
        n_class = S.shape[0]
        # 1. \forall i, j = 0, 1, 2, ..., n_class-1, i != j, S[i,j] --> [min_sim, __alpha]
        s_arr = np.unique(S)
        min_s = s_arr[0]
        max_s = s_arr[-2]  # s_arr[-1] = 1.0 = S[i, i]
        min_s_ = min_sim
        max_s_ = alpha
        for i in range(n_class):
            for j in range(i + 1, n_class):
                sij_ = Tool.LinearFunction(x1=min_s, y1=min_s_, x2=max_s, y2=max_s_, x=S[i, j])
                S[i, j] = sij_
                S[j, i] = sij_

        # 2. Compute Lower, Upper bound
        s_arr = list(np.unique(S))  # s_arr[0]==min_sim, s_arr[-1]= 1.0
        s_arr.append(0.0)  # s_arr[-1] == 0.0
        LB = np.zeros(shape=[n_class, n_class])
        UB = np.zeros(shape=[n_class, n_class])
        for i in range(n_class):
            LB[i, i] = beta
            UB[i, i] = 1.0
            for j in range(i + 1, n_class):
                sij = S[i, j]
                ind = s_arr.index(sij)
                sij_ = s_arr[ind - 1]  # if ind == 0, then  s_arr[ind - 1] == s_arr[- 1] == 0.0
                LB[i, j] = sij_
                LB[j, i] = sij_
                UB[i, j] = sij
                UB[j, i] = sij

        return {'LB': LB, 'UB': UB}

    @staticmethod
    def ToSimRank(S: np.array, num_grade=np.inf):
        """

        :param S: shape=[n, n], S=S.T, S[i,i]=1.0, S[i,j]<1.0
        :param num_grade:
        :return:
        """
        s_arr = np.unique(S)  # ascending order
        sim_rank_map = {}
        if s_arr.shape[0] <= num_grade:
            for r in range(s_arr.shape[0]):
                sim_rank_map[s_arr[r]] = r
        else:
            sim_rank_map[s_arr[-1]] = num_grade-1

            s_arr = s_arr[0: s_arr.shape[0]-1]
            len = s_arr.shape[0] // (num_grade-1)
            len_arr = len * np.ones(shape=[num_grade-1], dtype=np.int64)
            rest = s_arr.shape[0] - len * (num_grade-1)
            if rest > 0:
                len_arr[0:rest] += 1

            if np.sum(len_arr) != s_arr.shape[0]:
                raise Exception('np.sum(len_arr)=' + str(np.sum(len_arr)) + ' != s_arr.shape[0]=' + str(s_arr.shape[0]))

            beg_i = 0
            for r in range(num_grade-1):  # r=0, 1, 2, ..., num_grade-2
                end_i = beg_i + len_arr[r]
                for i in range(beg_i, end_i):
                    sim_rank_map[s_arr[i]] = r
                beg_i = end_i

        n = S.shape[0]
        R = np.zeros(shape=S.shape, dtype=np.int64)
        for i in range(n):
            for j in range(i, n):
                r = sim_rank_map[S[i, j]]
                R[i, j] = r
                R[j, i] = r
        return R

    @staticmethod
    def GenerateRandomSimilarity(max_class_id, rand_seed=123):
        np.random.seed(rand_seed)
        S = np.random.rand(max_class_id + 1, max_class_id + 1)
        S = 0.5 * (S + np.transpose(S))
        for i in range(S.shape[0]):
            S[i, i] = 1.0
        return S

    @staticmethod
    def Save2File(S: np.array, class_name_list, file_name):
        """

        :param S: shape=[n_class, n_class]
        :param class_name_list: shape=[n_class]
        :param file_name: string
        :return:
        """
        f = open(file=file_name, mode='w')
        f.write('ID-1\tID-2\tClass-1\tClass-2\tSim\n')
        n_cla = S.shape[0]
        for i in range(n_cla):
            for j in range(i, n_cla):
                f.write(str(i) + '\t' + str(j) + '\t' + class_name_list[i] + '\t' + class_name_list[j] + '\t'
                        + str(S[i, j]) + '\n')
        f.close()
