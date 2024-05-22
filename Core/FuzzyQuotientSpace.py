import random
from copy import copy
import hdf5storage
import numpy as np

from Core.QuotientSpace import Relation, Partition
from Core.Kit import Tool

"""
F: Fuzzy
Q: Quotient
S: Space

Sim: Similarity
Equ: Equivalent
Rel: Relation
Par: Partition
Hie: Hierarchical
Finer
Coarser
"""


class FuzzyRelation(object):
    def __init__(self, mat):
        super(FuzzyRelation, self).__init__()
        self.mat = mat

    def CutRelation(self, cut_value):
        return Relation(self.mat >= cut_value)

    def ToStr(self):
        R = self.mat.shape[0]
        C = self.mat.shape[1]
        rep_str = ''
        for r in range(R):
            for c in range(C):
                rep_str += str(self.mat[r, c]) + '\t'
            rep_str += '\n'
        return rep_str

    def Show(self):
        print(str(type(self)) + '=\n' + self.ToStr())

    def Save(self, file_name):
        f_writer = open(file=file_name, mode='w')
        f_writer.write(str(type(self)) + '=\n' + self.ToStr())
        f_writer.close()

    def Save_Mat(self, file_name):
        hdf5storage.savemat(file_name=file_name, mdict={'mat': self.mat})

    def ToHierarchicalPartition(self):
        vla_arr = np.unique(self.mat)  # ascending order
        par_list = []
        lambda_list = []
        for val in vla_arr:
            par = Relation.SimilarityRelation2Partition(bool_mat=self.mat >= val)
            par_list.append(par)
            lambda_list.append(val)
        return HierarchicalPartition(partition_list=par_list)

    def Scale_FSR(self, min_s, max_s):
        """

        :param min_s:
        :param max_s:
        :return:
        """
        space = np.unique(self.mat)
        min_v = space[0]
        max_v = space[-2]  # space[-1] == self.mat[i,i] == 1.0
        sam_num = self.mat.shape[0]
        for i in range(sam_num):
            for j in range(i+1, sam_num):
                s = Tool.LinearFunction(x1=min_v, y1=min_s, x2=max_v, y2=max_s, x=self.mat[i, j])
                self.mat[i, j] = s
                self.mat[j, i] = s

    def TransitiveClosure(self):
        M = copy(self)
        M2 = FuzzyRelation.Composition(FR1=M, FR2=M)
        while not FuzzyRelation.IsEqual(FR1=M, FR2=M2):
            M = M2
            M2 = FuzzyRelation.Composition(FR1=M, FR2=M)
        return M

    def ToLowerUpperBound(self, alpha=None, beta=0.8):
        """

        :return:
        """
        sam_num = self.mat.shape[0]
        LB = np.inf * np.ones(shape=[sam_num, sam_num])
        val_list = list(np.unique(self.mat))  # ascending order
        val_list.append(0.0)  # let val_arr[-1] equal to 0
        for r in range(sam_num):
            for c in range(r+1, sam_num):  # NOTE: c == r+1
                ind = val_list.index(self.mat[r, c])
                LB[r, c] = val_list[ind - 1]  # when ind=0, val_arr[ind - 1] == val_arr[-1]==0.0
                LB[c, r] = val_list[ind - 1]
            if beta is not None:
                LB[r, r] = beta

        UB = np.copy(self.mat)
        return LB, UB

    def CoarseningMultiValue2One(self, m=2):
        """
        Merge m adjacent values into one value
        :param m:
        :return:
        """
        R = self.mat.shape[0]
        C = self.mat.shape[1]
        if R != C:
            raise Exception('The # of rows != The # of columns.')

        mat = np.copy(self.mat)
        val_arr = []
        for r in range(R):
            for c in range(r + 1, R):
                val_arr.append(mat[r, c])
                val_arr.append(mat[c, r])
        val_arr = np.unique(np.array(val_arr))  # ascending order
        if val_arr.shape[0] < m:
            raise Exception('val_arr.shape[0] must >= m.')

        par = []
        equ_cla = []
        for val in val_arr:
            equ_cla.append(val)
            if equ_cla.__len__() == m:
                par.append(equ_cla)
                equ_cla = []
        if equ_cla.__len__() > 0:
            last_equ_cla = par[-1]
            last_equ_cla += equ_cla
            par[-1] = last_equ_cla

        old_new_val_map = {}
        for equ_cla in par:
            new_val = np.mean(np.array(equ_cla))
            for old_val in equ_cla:
                old_new_val_map[old_val] = new_val
        # print(old_new_val_map)
        for r in range(R):
            for c in range(r + 1, R):
                mat[r, c] = old_new_val_map[mat[r, c]]
                mat[c, r] = old_new_val_map[mat[c, r]]

        return FuzzyRelation(mat=mat)

    def GetValueDomain(self):
        return np.unique(self.mat)

    @staticmethod
    def CutIsTransitive(FR, cut_val_list):
        for cut_value in cut_val_list:
            rel = FR.CutRelation(cut_value=cut_value)
            if not Relation.IsTransitive(Rel=rel):
                FR.Show()
                rel.Show()
                # raise Exception('cut_value= ' + str(cut_value) + ', is not transitive.')
                return False
        return True

    @staticmethod
    def IsSameSort(FR1, FR2):
        R1 = FR1.mat.shape[0]
        C1 = FR1.mat.shape[1]
        R2 = FR2.mat.shape[0]
        C2 = FR2.mat.shape[1]

        if R1 == R2 and C1 == C2:
            pass
        else:
            raise Exception('FR1.shape must equal to FR2.shape')

        m1 = np.reshape(a=FR1.mat, newshape=[R1 * C1])
        m2 = np.reshape(a=FR2.mat, newshape=[R1 * C1])
        I1 = np.argsort(m1)
        I2 = np.argsort(m2)

        return np.sum(np.abs(I1 - I2)) == 0

    @staticmethod
    def Composition(FR1, FR2):
        R1 = FR1.mat.shape[0]
        C1 = FR1.mat.shape[1]
        R2 = FR2.mat.shape[0]
        C2 = FR2.mat.shape[1]
        if C1 != R2:
            raise Exception('# FR1.columns must equal to # FR2.rows')

        FR3_mat = np.zeros(shape=[R1, C2])
        for r in range(R1):
            for c in range(C2):
                row = FR1.mat[r, :]
                col = FR2.mat[:, c]
                FR3_mat[r, c] = np.max(np.minimum(row, col))

        return FuzzyRelation(mat=FR3_mat)

    @staticmethod
    def IsIsomorphic(FSR1, FSR2):
        hie_par1 = FSR1.ToHierarchicalPartition()
        hie_par2 = FSR2.ToHierarchicalPartition()
        return HierarchicalPartition.IsEqual(hie_par1=hie_par1, hie_par2=hie_par2)

    @staticmethod
    def IsIsomorphic_IntervalRegressionLoss(FSR_small, FSR_big):
        val_arr = np.unique(FSR_big.mat)
        val_list = list(val_arr)
        val_list.append(0)
        for i in range(val_arr.shape[0]):
            Rs = FSR_small.CutRelation(val_list[i - 1])
            Rb = FSR_big.CutRelation(val_list[i])
            if not Relation.IsEqual(Rel1=Rs, Rel2=Rb):
                return False

        return True

    @staticmethod
    def IsEqual(FR1, FR2):
        R1 = FR1.mat.shape[0]
        R2 = FR2.mat.shape[0]
        if R1 != R2:
            return False
        C1 = FR1.mat.shape[1]
        C2 = FR2.mat.shape[1]
        if C1 != C2:
            return False

        diff = np.abs(FR1.mat - FR2.mat)

        return np.sum(diff) == 0.0

    @staticmethod
    def IsSubsetEq(FR_small, FR_big):
        """
        FR_small[i,j] <= FR_big[i, j]
        :param FR_small:
        :param FR_big:
        :return:
        """
        R1 = FR_small.mat.shape[0]
        R2 = FR_big.mat.shape[0]
        if R1 != R2:
            return False
        C1 = FR_small.mat.shape[1]
        C2 = FR_big.mat.shape[1]
        if C1 != C2:
            return False

        for r in range(R1):
            for c in range(C1):
                if not FR_small.mat[r, c] <= FR_big.mat[r, c]:
                    return False

        return True

    @staticmethod
    def IsSubset(FR_small, FR_big):
        """
        FR_small[i,j] < FR_big[i, j]
        :param FR_small:
        :param FR_big:
        :return:
        """
        R1 = FR_small.mat.shape[0]
        R2 = FR_big.mat.shape[0]
        if R1 != R2:
            return False
        C1 = FR_small.mat.shape[1]
        C2 = FR_big.mat.shape[1]
        if C1 != C2:
            return False

        for r in range(R1):
            for c in range(C1):
                if not FR_small[r, c].mat < FR_big[r, c].mat:
                    return False

        return True

    @staticmethod
    def IsReflexive(FR):
        R = FR.mat.shape[0]
        C = FR.mat.shape[1]
        if R != C:
            return False
        for i in range(R):
            if FR.mat[i, i] != 1.0:
                return False
        return True

    @staticmethod
    def IsSymmetric(FR):
        R = FR.mat.shape[0]
        C = FR.mat.shape[1]
        if R != C:
            return False
        for r in range(R):
            for c in range(r + 1, C):
                if FR.mat[r, c] != FR.mat[c, r]:
                    return False
        return True

    @staticmethod
    def IsTransitive(FR):
        R = FR.mat.shape[0]
        C = FR.mat.shape[1]
        if R != C:
            return False
        FR_2 = FuzzyRelation.Composition(FR1=FR, FR2=FR)
        return FuzzyRelation.IsSubsetEq(FR_small=FR_2, FR_big=FR)

    @staticmethod
    def IsFuzzySimilarityRelation(FR):
        return FuzzyRelation.IsReflexive(FR) and \
               FuzzyRelation.IsSymmetric(FR)

    @staticmethod
    def IsFuzzyEquivalentRelation(FR):
        return FuzzyRelation.IsReflexive(FR) and \
               FuzzyRelation.IsSymmetric(FR) and \
               FuzzyRelation.IsTransitive(FR)

    @staticmethod
    def RandFSR_based_on_FER(FER):
        """

        :param FER: Reflexive and Symmetric and Transitive
        :return: FSR: Reflexive and Symmetric
        """
        R = FER.mat.shape[0]
        LB, UB = FER.ToLowerUpperBound()
        rand_mat = np.zeros(shape=[R, R])
        for r in range(R):
            rand_mat[r, r] = 1.0  # Reflexive
            for c in range(r + 1, R):
                rand_val = 0.0
                while rand_val == 0.0:
                    rand_val = np.random.rand()  # NOTE: np.random.rand() \in [0, 1)
                val = LB[r, c] + (UB[r, c] - LB[r, c]) * rand_val
                rand_mat[r, c] = val  # Symmetric
                rand_mat[c, r] = val  # Symmetric
        return FuzzyRelation(mat=rand_mat)

    @staticmethod
    def RandFSR_based_on_FSR(FSR):
        """

        :param FSR: Reflexive and Symmetric
        :return: FSR: Reflexive and Symmetric
        """
        R = FSR.mat.shape[0]
        LB, UB = FSR.ToLowerUpperBound()
        rand_mat = np.zeros(shape=[R, R])
        for r in range(R):
            rand_mat[r, r] = 1.0  # Reflexive
            for c in range(r + 1, R):
                rand_val = 0.0
                while rand_val == 0.0:
                    rand_val = np.random.rand()  # NOTE: np.random.rand() \in [0, 1)
                val = LB[r, c] + (UB[r, c] - LB[r, c]) * rand_val
                rand_mat[r, c] = val  # Symmetric
                rand_mat[c, r] = val  # Symmetric
        return FuzzyRelation(mat=rand_mat)

    @staticmethod
    def RandFR_based_on_FER(FER):
        """

        :param FER: Reflexive and Symmetric and Transitive
        :return: FR: Do not need to satisfy Reflexivity and Symmetry
        """
        R = FER.mat.shape[0]
        LB, UB = FER.ToLowerUpperBound()
        rand_mat = np.zeros(shape=[R, R])
        for r in range(R):
            for c in range(0, R):
                rand_val = 0.0
                while rand_val == 0.0:
                    rand_val = np.random.rand()  # NOTE: np.random.rand() \in [0, 1)
                val = LB[r, c] + (UB[r, c] - LB[r, c]) * rand_val
                rand_mat[r, c] = val
        return FuzzyRelation(mat=rand_mat)

    @staticmethod
    def Generate_Rand_FSR(sam_num, rand_seed=None):
        if rand_seed is not None:
            random.seed(rand_seed)
            np.random.seed(rand_seed)

        mat = np.random.rand(sam_num, sam_num)
        mat = 0.5 * (mat + np.transpose(mat))
        for i in range(sam_num):
            mat[i, i] = 1.0

        return FuzzyRelation(mat=mat)


class HierarchicalPartition(object):
    """
    An ordered sequence of partitions
    Coarsest Partition: {{0, 1, 2, ..., n}}
    Finest Partition:   {{0}, {1}, {2}, ..., {n}}
    """

    def __init__(self, partition_list):
        """

        :param partition_list:
               partition_list[0]: Coarsest Partition: {{0, 1, 2, ..., n}}
               partition_list[i]: Partition: {????}
               partition_list[-1]: Finest Partition:   {{0}, {1}, {2}, ..., {n}}
        :param lambda_list:
        :param min_lambda: threshold at Coarsest Partition
        :param __alpha: threshold at 2nd Finest Partition
        :param __beta: lower bound at Finest Partition
        """
        super(HierarchicalPartition, self).__init__()
        self.partition_list = partition_list
        self.__RemoveDuplicationPartition()

    def __RemoveDuplicationPartition(self):
        odl_par_list = self.partition_list
        old_layer_num = odl_par_list.__len__()
        self.partition_list = []
        last_equ_cla_num = 0
        for ind in range(old_layer_num):
            par = odl_par_list[ind]
            equ_cla_num = par.GetEquClaNum()
            if equ_cla_num > last_equ_cla_num:
                self.partition_list.append(par)
                last_equ_cla_num = equ_cla_num

    def Compute_lambda_list(self, finest_par_lambda=1.0, second_par_lambda=0.2, coarsest_par_lambda=1e-3):
        """

        :param finest_par_lambda: lambda for finest partition
        :param second_par_lambda: lambda for second-fine partition
        :param coarsest_par_lambda: lambda for coarsest partition
        :return:
        """
        lambda_list = []
        min_equ_cla_num = self.partition_list[0].GetEquClaNum()
        second_equ_cla_num = self.partition_list[-1].GetEquClaNum()
        par_num = self.GetLayerNum()
        for ind in range(par_num - 1):
            equ_cla_num = self.partition_list[ind].GetEquClaNum()
            lam = Tool.LinearFunction(x1=second_equ_cla_num, y1=second_par_lambda,
                                      x2=min_equ_cla_num, y2=coarsest_par_lambda,
                                      x=equ_cla_num)
            lambda_list.append(lam)
        lambda_list.append(finest_par_lambda)
        return lambda_list

    def GetLayerNum(self):
        return self.partition_list.__len__()

    def GetPartition(self, layer_ind):
        return self.partition_list[layer_ind]

    def ToFuzzyEquivalentRelation(self, finest_par_lambda=1.0, second_par_lambda=0.2,
                                  coarsest_par_lambda=1e-3) -> FuzzyRelation:
        lambda_list = self.Compute_lambda_list(finest_par_lambda=finest_par_lambda, second_par_lambda=second_par_lambda,
                                               coarsest_par_lambda=coarsest_par_lambda)
        par_num = self.partition_list.__len__()
        p_list = list(range(par_num))
        p_list.reverse()
        par = self.partition_list[0]
        sam_num = max(par.GetSampleList()) + 1
        mat = np.ones(shape=[sam_num, sam_num])
        for i in range(sam_num):
            for j in range(i + 1, sam_num):
                for p in p_list:
                    par = self.partition_list[p]
                    if par.IsInSameEquCla(e1=i, e2=j):
                        mat[i, j] = lambda_list[p]
                        mat[j, i] = lambda_list[p]
                        break
        return FuzzyRelation(mat=mat)

    def ToSimilarityRank(self) -> np.array:
        """

        :return:
        """
        par = self.partition_list[0]
        sam_num = max(par.GetSampleList()) + 1
        mat = -np.inf * np.ones(shape=[sam_num, sam_num])

        par_num = self.partition_list.__len__()
        p_list = list(range(par_num))
        p_list.reverse()  # [par_num-1, par_num-2, ..., 1, 0]
        for i in range(sam_num):
            mat[i, i] = par_num-1
            for j in range(i + 1, sam_num):
                for p in p_list:
                    par = self.partition_list[p]
                    if par.IsInSameEquCla(e1=i, e2=j):
                        mat[i, j] = p
                        mat[j, i] = p
                        break
        return mat

    def ToStr(self):
        if self.partition_list.__len__() == 0:
            return str(type(self)) + '= [null]'
        rep_str = str(type(self)) + '=[\n'
        par = self.partition_list[0]
        rep_str += 'Layer=\t1\t' + par.ToStr()
        for ind in range(1, self.partition_list.__len__()):
            par = self.partition_list[ind]
            rep_str += '\n'
            rep_str += 'Layer=\t' + str(ind + 1) + '\t' + par.ToStr()
        rep_str += '\n]'
        return rep_str

    def Show(self):
        print(self.ToStr())

    def Save(self, file_name):
        f_writer = open(file=file_name, mode='w')
        f_writer.write(self.ToStr())
        f_writer.close()

    @staticmethod
    def FromTextFile(file_name):
        """

        :param file_name: e.g.,
        {{0,1,2}}
        {{0,1},{2}}
        {{0},{1},{2}}
        :return:
        """
        f_reader = open(file=file_name, mode='r')
        par_list = []
        line = f_reader.readline()
        while line != '':
            par = Partition.FromStr(par_str=line.replace('\n', ''))  # delete the '\n'
            par_list.append(par)
            line = f_reader.readline()

        f_reader.close()
        return HierarchicalPartition(partition_list=par_list)

    @staticmethod
    def IsEqual(hie_par1, hie_par2):
        layer_num = hie_par1.GetLayerNum()
        if layer_num != hie_par2.GetLayerNum():
            return False

        for layer in range(layer_num):
            par1 = hie_par1.partition_list[layer]
            par2 = hie_par2.partition_list[layer]
            if not Partition.IsEqual(par1=par1, par2=par2):
                return False

        return True

    @staticmethod
    def IsProperSubset(big_hie_par, small_hie_par):
        layer_num = big_hie_par.GetLayerNum()
        if layer_num <= small_hie_par.GetLayerNum():
            return False

        big_ind = 0
        par_list = []
        for par1 in small_hie_par.partition_list:
            equ_cla_num = par1.GetEquClaNum()
            has = False
            for ind in range(big_ind, layer_num):
                par2 = big_hie_par.partition_list[ind]
                if equ_cla_num == par2.GetEquClaNum():
                    par_list.append(par2)
                    big_ind = ind + 1
                    has = True
                    break
            if not has:
                return False
        hie_par = HierarchicalPartition(partition_list=par_list)
        return HierarchicalPartition.IsEqual(hie_par1=hie_par, hie_par2=small_hie_par)

    @staticmethod
    def GenerateExample():
        par1 = Partition(equ_cla_list=[[0, 1, 2, 3, 4, 5, 6]])
        par2 = Partition(equ_cla_list=[[0, 1, 2, 3, 4, 5], [6]])
        par3 = Partition(equ_cla_list=[[0, 1, 2], [3, 4, 5], [6]])
        par4 = Partition(equ_cla_list=[[0, 1], [2], [3, 4, 5], [6]])
        par5 = Partition(equ_cla_list=[[0], [1], [2], [3], [4], [5], [6]])
        par_list = [par1, par2, par3, par4, par5]
        hie_par = HierarchicalPartition(partition_list=par_list)
        hie_par.lambda_list = [0.2, 0.4, 0.6, 0.8, 1.0]
        return hie_par
