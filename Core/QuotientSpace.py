import numpy as np
from Core.Kit import List

"""
Q: Quotient
S: Space
Sim: Similarity
Equ: Equivalent
Rel: Relation
Par: Partition
"""


class Relation(object):
    def __init__(self, bool_mat):
        super(Relation, self).__init__()
        self.bool_mat = bool_mat

    def ToStr(self):
        R = self.bool_mat.shape[0]
        C = self.bool_mat.shape[1]
        rep_str = str(type(self)) + '=[\n'
        for r in range(R):
            rwo_str = ''
            if self.bool_mat[r, 0]:
                rwo_str += '1'
            else:
                rwo_str += '0'
            for c in range(C):
                if self.bool_mat[r, c]:
                    rwo_str += '\t1'
                else:
                    rwo_str += '\t0'
            rwo_str += '\n'

            rep_str += rwo_str
        rep_str += ']'
        return rep_str

    def Show(self):
        print(self.ToStr())

    @staticmethod
    def IsReflexive(Rel):
        R = Rel.bool_mat.shape[0]
        C = Rel.bool_mat.shape[1]
        if R != C:
            return False
        for i in range(R):
            if not Rel.bool_mat.mat[i, i]:
                return False
        return True

    @staticmethod
    def IsSymmetric(Rel):
        R = Rel.mat.shape[0]
        C = Rel.mat.shape[1]
        if R != C:
            return False
        for r in range(R):
            for c in range(r + 1, C):
                if Rel.mat[r, c] != Rel.mat[c, r]:
                    return False
        return True

    @staticmethod
    def IsTransitive(Rel):
        R = Rel.bool_mat.shape[0]
        C = Rel.bool_mat.shape[1]
        if R != C:
            return False
        Rel_2 = Relation.Composition(Rel1=Rel, Rel2=Rel)
        return Relation.IsEqual(Rel1=Rel, Rel2=Rel_2)

    @staticmethod
    def IsEqual(Rel1, Rel2):
        R1 = Rel1.bool_mat.shape[0]
        R2 = Rel2.bool_mat.shape[0]
        if R1 != R2:
            return False

        C1 = Rel1.bool_mat.shape[1]
        C2 = Rel2.bool_mat.shape[1]
        if C1 != C2:
            return False

        # for r in range(R1):
        #     for c in range(C1):
        #         if Rel1.bool_mat[r, c] != Rel2.bool_mat[r, c]:
        #             return False
        #
        # return True

        diff = np.abs((Rel1.bool_mat + 0.0) - (Rel2.bool_mat + 0.0))
        return np.sum(diff) == 0.0

    @staticmethod
    def Union(Rel1, Rel2):
        R1 = Rel1.bool_mat.shape[0]
        R2 = Rel2.bool_mat.shape[0]
        if R1 != R2:
            raise Exception('Rel1.shape must equal to Rel2.shape.')

        C1 = Rel1.bool_mat.shape[1]
        C2 = Rel2.bool_mat.shape[1]
        if C1 != C2:
            raise Exception('Rel1.shape must equal to Rel2.shape.')

        mat = np.zeros(shape=[R1, C1]) == 1.0
        for r in range(R1):
            for c in range(C1):
                mat = Rel1.bool_mat[r, c] or Rel2.bool_mat[r, c]

        return Relation(bool_mat=mat)

    @staticmethod
    def Composition(Rel1, Rel2):
        R1 = Rel1.bool_mat.shape[0]
        C1 = Rel1.bool_mat.shape[1]
        R2 = Rel2.bool_mat.shape[0]
        C2 = Rel2.bool_mat.shape[1]
        if C1 != R2:
            raise Exception('# Rel1.columns must equal to # Rel2.rows')

        mat = np.zeros(shape=[R1, C2]) == 1.0
        for r in range(R1):
            for c in range(C2):
                val = False
                for k in range(C1):
                    val = val or (Rel1.bool_mat[r, k] and Rel2.bool_mat[k, c])
                mat[r, c] = val
        return Relation(bool_mat=mat)

    @staticmethod
    def TransitiveClosure(Rel):
        t_rel = Relation(bool_mat=np.copy(Rel.bool_mat))
        rel_power = Relation(bool_mat=np.copy(Rel.bool_mat))
        while True:
            old_t_rel = Relation(bool_mat=np.copy(t_rel.bool_mat))
            rel_power = Relation.Composition(Rel1=rel_power, Rel2=Rel)
            t_rel = Relation.Union(Rel1=rel_power, Rel2=t_rel)
            if Relation.IsEqual(Rel1=t_rel, Rel2=old_t_rel):
                break
        return t_rel

    @staticmethod
    def __IndexOfTrue(bool_list, beg_ind=0):
        for i in range(beg_ind, bool_list.__len__()):
            if bool_list[i]:
                return i
        return None

    @staticmethod
    def SimilarityRelation2Partition(bool_mat):
        """

        :param bool_mat: Reflexive and Symmetric and Do Not need to be Transitive
        :return:
        """
        R = bool_mat.shape[0]
        C = bool_mat.shape[1]
        if R != C:
            raise Exception('# of rows must equal to # of columns.')
        is_rest = [True] * R
        equ_cla_list = []
        seed_sam = Relation.__IndexOfTrue(bool_list=is_rest)
        while seed_sam is not None:
            queue = List()
            queue.InQueue(seed_sam)
            is_rest[seed_sam] = False
            equ_cla = []
            while not queue.IsEmpty():
                sam = queue.OutQueue()
                equ_cla.append(sam)
                for i in range(R):
                    if is_rest[i] and bool_mat[sam, i]:
                        queue.InQueue(ele=i)
                        is_rest[i] = False
            equ_cla_list.append(equ_cla)
            seed_sam = Relation.__IndexOfTrue(bool_list=is_rest, beg_ind=seed_sam + 1)
        return Partition(equ_cla_list=equ_cla_list)


class Partition(object):
    """
    set of equivalent classes
    """

    def __init__(self, equ_cla_list):
        """

        :param equ_cla_list: 2-D list
        """
        super(Partition, self).__init__()
        self.equ_cla_list = equ_cla_list
        self.__sort()

    def __sort(self):
        equ_cla_min = []
        for euq_cla in self.equ_cla_list:
            equ_cla_min.append(min(euq_cla))
        equ_cla_min = np.array(equ_cla_min)
        sort_ind = np.argsort(equ_cla_min)
        old_equ_cla_list = self.equ_cla_list
        self.equ_cla_list = []
        for ind in sort_ind:
            equ_cla = np.array(old_equ_cla_list[ind])
            equ_cla = list(np.sort(equ_cla))
            self.equ_cla_list.append(equ_cla)

    def __len__(self):
        return self.equ_cla_list.__len__()

    def GetEquClaNum(self):
        return self.equ_cla_list.__len__()

    @staticmethod
    def __EquCla2Str(equ_cla):
        """

        :param equ_cla: list, int
        :return:
        """
        if equ_cla.__len__() == 0:
            return '{}'

        str1 = '{' + str(equ_cla[0])
        for i in range(1, equ_cla.__len__()):
            str1 += (', ' + str(equ_cla[i]))
        str1 += '}'
        return str1

    def ToStr(self):
        if self.equ_cla_list.__len__() == 0:
            return str(type(self)) + '= {{}}'

        rep_str = str(type(self)) + '=\t{' + Partition.__EquCla2Str(self.equ_cla_list[0])
        for ind in range(1, self.equ_cla_list.__len__()):
            rep_str += ', '
            rep_str += Partition.__EquCla2Str(self.equ_cla_list[ind])
        rep_str += '}'
        return rep_str

    def GetSampleList(self):
        sam_list = []
        for equ_cla in self.equ_cla_list:
            sam_list += equ_cla
        return sam_list

    def IsInSameEquCla(self, e1, e2):
        for equ_cla in self.equ_cla_list:
            if equ_cla.__contains__(e1):
                if equ_cla.__contains__(e2):
                    return True
                else:
                    return False

    def Check(self):
        # print('Partition.Check()')
        sam_list = self.GetSampleList()
        sam_arr = np.array(sam_list)
        sam_arr_sort = np.unique(sam_arr)
        if sam_arr_sort.shape[0] != sam_arr.shape[0]:
            raise Warning('There are duplicate elements in the partition.')

        for i in range(sam_arr_sort.shape[0]):
            if i != sam_arr_sort[i]:
                raise Warning('i= ' + str(i) + ', but sam_arr_sort[i]= ' + str(sam_arr_sort[i]))

    def To01Matrix(self):
        sam_list = self.GetSampleList()
        sam_num = max(sam_list) + 1
        mat = np.zeros(shape=[sam_num, sam_num])
        for equ_cla in self.equ_cla_list:
            r_arr = np.array(equ_cla)
            rc_arr = np.meshgrid(r_arr, r_arr)
            mat[rc_arr[0], rc_arr[1]] = 1.0
            mat[rc_arr[1], rc_arr[0]] = 1.0
        return mat

    def Show(self):
        print(self.ToStr())

    @staticmethod
    def IsEqual(par1, par2):
        if par1.GetEquClaNum() != par2.GetEquClaNum():
            return False

        sam_arr1 = np.array(par1.GetSampleList())
        sam_arr2 = np.array(par2.GetSampleList())
        if sam_arr1.shape[0] != sam_arr2.shape[0]:
            return False

        sam_arr1 = np.sort(sam_arr1)
        sam_arr2 = np.sort(sam_arr2)
        diff = np.abs(sam_arr1 - sam_arr2)
        if np.sum(diff) != 0.0:
            return False

        mat1 = par1.To01Matrix()
        mat2 = par2.To01Matrix()
        diff = np.abs(mat1 - mat2)
        if np.sum(diff) != 0.0:
            return False

        return True

    @staticmethod
    def FromStr(par_str):
        """

        :param par_str: e.g., '{{0,1,2},{3,4},{5,6,7}}'
        :return:
        """
        par_str = par_str[2:len(par_str) - 2]
        equ_cla_list = []
        equ_cla_str_list = par_str.split('},{')
        for equ_cla_str in equ_cla_str_list:
            equ_cla = []
            sam_str_list = equ_cla_str.split(',')
            for sam_str in sam_str_list:
                equ_cla.append(int(sam_str))
            equ_cla_list.append(equ_cla)

        return Partition(equ_cla_list=equ_cla_list)
