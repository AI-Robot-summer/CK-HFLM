import os
from copy import copy
import hdf5storage
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torchvision.io import image
from torchvision import transforms
from torch.utils.data.dataset import T_co
from nltk.corpus import wordnet as wn

from Core.FuzzyQuotientSpace import HierarchicalPartition, FuzzyRelation
from Core.Kit import Tool

"""
FER: Fuzzy Equivalent Relation
LB: Lower Bound
UB: Upper Bound
Mat: matrix

sam: sample
cla: class
num: number
Inf: information
"""


class ClassSpaceInformation(object):
    """
    class_id: int, the id of class, must belong to {0, 1, ..., cla_num-1}

    class_folder_name:
    Each class corresponds to a folder,
    There are many files in one folder,
    Each file corresponds to a sample (image)


    In WordNet, a synset is identified with a 3-part name of the form:
    word.pos.nn, e.g. dog.n.01
    pos in {NOUN, ADJ, ADV}, {n, v, adj, adv}

    class_word: string, the name of class, the word of WordNet synset, e.g. 'dog'
    class_pos: string, the pos of WordNet synset, e.g. 'n'
    class_nn: string, the nn of WordNet synset, e.g. '01'

    other:
    """
    unknown_class_id = -999

    def __init__(self, cls_space_inf_txt_file):
        """

        :param cls_space_inf_txt_file: *.txt file,
        first line (Table head): class_id|class_folder_name|class_word|class_pos|class_nn|other
        other lines:
        """
        # class_id | class_folder_name | class_word | class_pos | class_nn | other
        super(ClassSpaceInformation, self).__init__()
        self.class_id_list = []
        self.class_folder_name_list = []
        self.class_word_list = []
        self.class_pos_list = []
        self.class_nn_list = []
        self.other_list = []

        self.class_folder_name_id_map = {}
        self.class_word_pos_nn_id_map = {}
        self.other_list_id_map = {}

        f_reader = open(file=cls_space_inf_txt_file, mode='r')
        _ = f_reader.readline()  # skip the first line
        line = f_reader.readline()
        while line != '':
            line = line.replace('\n', '')
            res = line.split(sep='|')
            class_id = int(res[0])
            class_folder_name = res[1]
            class_word = res[2]
            class_pos = res[3]
            class_nn = res[4]
            other = res[5]

            self.class_id_list.append(class_id)
            self.class_folder_name_list.append(class_folder_name)
            self.class_word_list.append(class_word)
            self.class_pos_list.append(class_pos)
            self.class_nn_list.append(class_nn)
            self.other_list.append(other)

            self.class_folder_name_id_map[class_folder_name] = class_id
            class_word_pos_nn = class_word + '.' + class_pos + '.' + class_nn
            self.class_word_pos_nn_id_map[class_word_pos_nn] = class_id
            self.other_list_id_map[other] = class_id

            line = f_reader.readline()

        f_reader.close()

    def id2folder_name(self, class_id):
        return self.class_folder_name_list[class_id]

    def folder_name2id(self, class_folder_name):
        return self.class_folder_name_id_map[class_folder_name]

    def GetClassID_list(self):
        return self.class_id_list

    def GetClassNum(self):
        return len(self.class_id_list)

    def Show(self):
        print(' class_id | class_folder_name | class_word | class_pos | class_nn | other')
        for i in range(self.GetClassNum()):
            print(str(self.class_id_list[i]) + '|' +
                  self.class_folder_name_list[i] + '|' +
                  self.class_word_list[i] + '|' +
                  self.class_pos_list[i] + '|' +
                  self.class_nn_list[i] + '|' +
                  self.other_list[i])

    def ToSynsetList(self):
        class_num = self.class_id_list.__len__()
        synset_list = []
        for i in range(class_num):
            synset_ = wn.synset(self.class_word_list[i]
                                + '.' + self.class_pos_list[i]
                                + '.' + self.class_nn_list[i])
            synset_list.append(synset_)
        return synset_list


class MyDataSet(Dataset):
    def __init__(self):
        super(MyDataSet, self).__init__()
        self._x = None  # np.array or list of string
        self._y = None  # np.array
        self._class_Sim_Mat_UB = np.inf * np.ones(shape=[100, 100])  # np.array
        self._class_Sim_Mat_LB = -np.inf * np.ones(shape=[100, 100])  # np.array

        self._batch_generator = None

    def __len__(self):
        return self._y.shape[0]

    def __preprocessing__(self, x: Tensor) -> Tensor:
        pass

    def __getitem__(self, index) -> T_co:
        y = torch.tensor(data=self._y[index])
        if type(self._x[0]) is str:
            x = self.__preprocessing__(image.read_image(self._x[index]))
        else:
            x = torch.tensor(data=self._x[index])
        return x, y

    def GetClassSmi_LUB(self):
        if type(self._class_Sim_Mat_LB) is np.ndarray:
            return {'LB': torch.tensor(data=self._class_Sim_Mat_LB),
                    'UB': torch.tensor(data=self._class_Sim_Mat_UB)}
        elif type(self._class_Sim_Mat_LB) is torch.Tensor:
            return {'LB': self._class_Sim_Mat_LB,
                    'UB': self._class_Sim_Mat_UB}
        else:
            return None

    def GetClassRelationInterval(self, class_label_arr):
        LB_Mat = self._class_Sim_Mat_LB[class_label_arr, :]
        LB_Mat = LB_Mat[:, class_label_arr]

        UB_Mat = self._class_Sim_Mat_UB[class_label_arr, :]
        UB_Mat = UB_Mat[:, class_label_arr]

        if type(self._class_Sim_Mat_LB) is np.ndarray:
            LB_Mat = torch.tensor(data=LB_Mat)
            UB_Mat = torch.tensor(data=UB_Mat)

        return {'LB_Mat': LB_Mat, 'UB_Mat': UB_Mat}

    def GetClassSimilarityMat(self) -> torch.Tensor:
        if type(self._class_Sim_Mat_LB) is np.ndarray:
            return torch.tensor(data=self._class_Sim_Mat_UB)
        elif type(self._class_Sim_Mat_UB) is torch.Tensor:
            return self._class_Sim_Mat_UB

    def GetClassNum(self):
        if type(self._y) == np.ndarray:
            return np.unique(self._y).shape[0]
        else:
            return torch.unique(self._y).shape[0]

    def GetMaxClassID(self):
        if type(self._y) == np.ndarray:
            return np.max(self._y)
        else:
            return torch.max(self._y)

    def GetSampleNum(self):
        return self._y.shape[0]

    def Get_x_y(self, index):
        y = torch.tensor(data=self._y[index])
        if type(self._x[0]) is str:
            x = image.read_image(self._x[index])
        else:
            x = torch.tensor(data=self._x[index])
        return x, y

    def GetSubsetOfClass(self, class_id):
        return self.GetSubset(sam_indices=self._y == class_id)

    def GetSubset(self, sam_indices: np.array):
        """

        :param sam_indices: shape=[_sam_num], dtype=int or bool
        :return:
        """
        data_set = MyDataSet()
        data_set._y = copy(self._y[sam_indices])
        data_set._class_Sim_Mat_UB = copy(self._class_Sim_Mat_UB)
        data_set._class_Sim_Mat_LB = copy(self._class_Sim_Mat_LB)
        if type(self._x[0]) is str:
            data_set._x = []
            if type(sam_indices[0]) is bool:
                for sam_ind in range(self.GetSampleNum()):
                    if sam_indices[sam_ind]:
                        data_set._x.append(self._x[sam_ind])
            else:
                for sam_ind in sam_indices:
                    data_set._x.append(self._x[sam_ind])
        elif type(self._x) is np.ndarray:
            data_set._x = copy(self._x[sam_indices])
        elif type(self._x) is torch.Tensor:
            data_set._x = copy(self._x[sam_indices])
        else:
            raise Exception('type(self._x) == ' + str(type(self._x)) + ' is not support.')
        return data_set

    def GetAll_x_y(self):
        y = torch.tensor(data=self._y)
        if type(self._x[0]) is str:
            x_list = []
            for i in range(self.GetSampleNum()):
                x_list.append(self.__getitem__(index=i))
            x = torch.cat(tensors=x_list, dim=0)
        elif type(self._x) is np.ndarray:
            x = torch.tensor(data=self._x)
        elif type(self._x) is Tensor:
            x = self._x
        return x, y

    def GerAll_y(self):
        if type(self._y) is np.ndarray:
            return self._y
        elif type(self._y) is torch.Tensor:
            return self._y.detach().clone().cpu().numpy()

    def GetBatchX_y(self, sam_indices):
        X = None
        y = None
        if type(self._x[0]) is str:
            X = []
            for index in sam_indices:
                X.append(image.read_image(self._x[index]))
            X = torch.cat(tensors=X, dim=0)
            y = torch.tensor(data=self._y[sam_indices])
        elif type(self._x) is np.ndarray:
            X = torch.tensor(data=self._x[sam_indices])
            y = torch.tensor(data=self._y[sam_indices])
        elif type(self._x) is torch.Tensor:
            X = self._x[sam_indices]
            y = self._y[sam_indices]
        return {'X': X, 'y': y}

    def ResetBatchGenerator(self):
        self._batch_generator = None

    def InitEpoch(self, batch_size, batch_class_num=None, epoch_num=1, do_shuffle=True):
        if self._batch_generator is None:
            if batch_class_num is None:
                self._batch_generator = Partition_BG(data_set=self)
            else:
                self._batch_generator = ClassSample_BG(data_set=self)

        self._batch_generator.InitEpoch(batch_size=batch_size, batch_class_num=batch_class_num, epoch_num=epoch_num,
                                        do_shuffle=do_shuffle)

    def NextBatch(self):
        return self._batch_generator.NextBatch()

    def GetSampleNum(self):
        return self._y.shape[0]

    def GetClassSampleNum(self):
        """

        :return:
        """
        num_arr = []
        ind_arr = []
        class_space = self.GetClassSpace()
        y = self.GerAll_y()
        ind_bank = np.arange(0, y.shape[0])
        for c in class_space:
            Ic = y == c
            ind_c = ind_bank[Ic]
            num_arr.append(ind_c.shape[0])
            ind_arr.append(ind_c)

        return np.array(num_arr), np.concatenate(ind_arr, axis=0)

    def SetClassFSR_LUB(self, alpha=0.2, beta=0.8):
        """
        When the class similarity matrix is set in this way,
        it degenerates into a conventional fuzzy learning machine (Fuzzy Permission Loss)
        :param alpha: 0.0 <= __alpha <= 0.5 <= __beta <= 1.0
        :param beta:
        :return:
        """
        class_num = np.max(self._y) + 1
        self._class_Sim_Mat_LB = np.zeros(shape=[class_num, class_num])
        self._class_Sim_Mat_UB = np.zeros(shape=[class_num, class_num])
        for i in range(class_num):
            self._class_Sim_Mat_LB[i, i] = beta
            self._class_Sim_Mat_UB[i, i] = 1.0
            for j in range(i + 1, class_num):
                self._class_Sim_Mat_LB[i, j] = 0.0
                self._class_Sim_Mat_LB[j, i] = 0.0
                self._class_Sim_Mat_UB[i, j] = alpha
                self._class_Sim_Mat_UB[j, i] = alpha

    def SetClassFSR_LUB2(self, LB, UB):
        self._class_Sim_Mat_LB = LB
        self._class_Sim_Mat_UB = UB

    def SetClassFSR3(self, classFSR):
        self._class_Sim_Mat_UB = classFSR

    def ComputeClassFSR_LUB(self, A: np.array, alpha, beta, min_sim=1e-3):
        """

        :param A: class attribute matrix, shape=[num_class, num_attribute]
        :param alpha:  0 <= __alpha <= 0.5 <= __beta <= 1.0
        :param beta:
        :param min_sim:
        :return:
        """
        n_class = A.shape[0]
        if self.GetClassNum() != n_class:
            raise Exception('H1.shape[0] != number of classes.')

        # 1. CosSimilarity
        d = np.sum(A * A, axis=1)
        d = np.sqrt(d)
        d = 1.0 / d
        D = np.diag(d)
        S = np.matmul(A, np.transpose(A))
        S = np.matmul(np.matmul(D, S), D)  # -1 <= S[i, j] <= 1, S[i, i] = 1.0

        # 2. [-1, 1] --> [0, 1]
        S = (S + 1.0) / 2.0  # 0 <= S[i, j] <= 1, S[i, i] = 1.0

        # 3. \forall i, j = 0, 1, 2, ..., n_class-1, i != j, S[i,j] --> [min_sim, __alpha]
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

        # 4. Compute Lower, Upper bound
        s_arr = list(np.unique(S))  # s_arr[0]==min_sim, s_arr[-1]= 1.0
        s_arr.append(0.0)  # s_arr[-1] == 0.0
        self._class_Sim_Mat_LB = np.zeros(shape=[n_class, n_class])
        self._class_Sim_Mat_UB = np.zeros(shape=[n_class, n_class])
        for i in range(n_class):
            self._class_Sim_Mat_LB[i, i] = beta
            self._class_Sim_Mat_UB[i, i] = 1.0
            for j in range(i + 1, n_class):
                sij = S[i, j]
                ind = s_arr.index(sij)
                sij_ = s_arr[ind - 1]  # if ind == 0, then  s_arr[ind - 1] == s_arr[- 1] == 0.0
                self._class_Sim_Mat_LB[i, j] = sij_
                self._class_Sim_Mat_LB[j, i] = sij_
                self._class_Sim_Mat_UB[i, j] = sij
                self._class_Sim_Mat_UB[j, i] = sij

    def GetClassSpace(self):
        if type(self._y) == np.ndarray:
            return np.unique(self._y)
        else:
            return torch.unique(self._y).detach().clone().cpu().numpy()

    def CheckClassSpace(self):
        if self._y is None:
            raise Warning('self._y is None!!')
        else:
            class_space = np.unique(self._y)
            for i in range(class_space.shape[0]):
                if i != class_space[i]:
                    raise Warning('i= ' + str(i) + ', but _class_space[i]= ' + str(class_space[i]))

    def GetSampleFileName(self, index):
        if type(self._x[0]) is str:
            return self._x[index]
        else:
            return str(type(self)) + '-Not-File-' + str(index)

    def ShowDataSetInformation(self):
        print(type(self))
        print('sample_num= ' + str(self.GetSampleNum()))
        print('class_num= ' + str(self.GetClassNum()))
        print('class-id, sample-number')
        class_space = np.unique(self._y)
        for class_id in class_space:
            print(str(class_id) + ', ' + str(np.sum(self._y == class_id)))

    def SaveDataSetInformation(self, file_name):
        f_writer = open(file=file_name, mode='w')
        f_writer.write(str(type(self)) + '\n')
        f_writer.write('sample_num= ' + str(self.GetSampleNum()) + '\n')
        f_writer.write('class_num= ' + str(self.GetClassNum()) + '\n')
        f_writer.write('class-id, sample-number\n')
        class_space = np.unique(self._y)
        for class_id in class_space:
            f_writer.write(str(class_id) + ', ' + str(np.sum(self._y == class_id)) + '\n')

        cla_space = self.GetClassSpace()

        f_writer.write('class FER and FER-lower-bound matrix=\n')
        f_writer.write('FER\t')
        for c in cla_space:
            f_writer.write('class ' + str(c) + '\t')
        f_writer.write('\n')

        for c1 in cla_space:
            f_writer.write('class ' + str(c1) + '\t')
            for c2 in cla_space:
                f_writer.write('(' + str(self._class_Sim_Mat_LB[c1, c2]) + ', ' +
                               str(self._class_Sim_Mat_UB[c1, c2]) + ')\t')
            f_writer.write('\n')
        f_writer.close()

    def SaveSampleInformation(self, file_name):
        f_writer = open(file=file_name, mode='w')
        # f_writer.write('sample-ID\tfile-name\timage-channel\timage-high\timage-width\tclass-id\n')
        f_writer.write('sample-ID\tfile-name\tsample-shape\tclass-id\n')
        for i in range(self.GetSampleNum()):
            if i % 1000 == 0:
                print(i)
            x, y = self.Get_x_y(i)
            f_writer.write(str(i) + '\t'
                           + self.GetSampleFileName(i) + '\t'
                           + str(x.shape) + '\t'
                           + str(y.item()) + '\n')

        f_writer.close()

    def Save2MatFile(self, file_name):
        hdf5storage.savemat(file_name=file_name,
                            mdict={'x': self._x, 'y': self._y,
                                   'class_FER_Mat': self._class_Sim_Mat_UB,
                                   'class_FER_Mat_LB': self._class_Sim_Mat_LB})

    @staticmethod
    def LoadFromMatFile(file_name):
        data = hdf5storage.loadmat(file_name=file_name)
        dataset = MyDataSet()
        dataset._x = data['x']
        dataset._y = data['y']
        dataset._class_Sim_Mat_UB = data['class_FER_Mat']
        dataset._class_Sim_Mat_LB = data['class_FER_Mat_LB']
        return dataset

    @staticmethod
    def UnionDataset(dataset_list):
        dataset = MyDataSet()
        x_list = []
        y_list = []
        if type(dataset_list[0]._x[0]) is str:
            for data in dataset_list:
                y_list.append(data._y)
                x_list += data._x
            dataset._x = x_list
            dataset._y = np.concatenate(y_list, axis=0)
        elif type(dataset_list[0]._x[0]) is np.ndarray:
            for data in dataset_list:
                y_list.append(data._y)
                x_list.append(data._x)
            dataset._x = np.concatenate(x_list, axis=0)
            dataset._y = np.concatenate(y_list, axis=0)
        elif type(dataset_list[0]._x[0]) is torch.Tensor:
            for data in dataset_list:
                y_list.append(data._y)
                x_list.append(data._x)
            dataset._x = torch.cat(tensors=x_list, dim=0)
            dataset._y = torch.cat(tensors=y_list, dim=0)
        else:
            raise Exception('type(MyDataSet._x) == ' + str(type(dataset_list[0]._x[0])) + ' is not support.')

        dataset._class_Sim_Mat_LB = copy(dataset_list[0]._class_Sim_Mat_LB)
        dataset._class_Sim_Mat_UB = copy(dataset_list[0]._class_Sim_Mat_UB)
        return dataset

    @staticmethod
    def ExtractFeature(net, data_set, batch_size=10000, device=torch.device('cpu')):
        data_set.InitEpoch(batch_size=batch_size, do_shuffle=False)
        H_list = []
        while True:
            batch_data = data_set.NextBatch()
            batch_X = batch_data['X'].to(device)
            batch_H = net(batch_X)
            H_list.append(batch_H.detach().clone().cpu())
            if batch_data['is_last_batch']:
                break

        H = torch.cat(tensors=H_list, dim=0)
        H = H.detach().clone().numpy()
        return H, data_set.GerAll_y()


class TableDataSet(MyDataSet):
    def __init__(self, Xy_mat_file=None):
        """

        :param Xy_mat_file: {'X': shape=[n_sample, n_feature], 'y': shape=[n_sample, 1]}
        """
        super(TableDataSet, self).__init__()
        if Xy_mat_file is not None:
            mat = hdf5storage.loadmat(file_name=Xy_mat_file)
            self._x = mat['X'].astype(np.float64)
            self._y = np.squeeze(mat['y']).astype(np.int64)
            self.CheckClassSpace()

    def SplitTrainTest(self, is_train):
        """

        :param is_train: shape=[n_sample], bool
        :return:
        """
        tr_data = TableDataSet()
        tr_data._x = self._x[is_train]
        tr_data._y = self._y[is_train]
        tr_data._class_Sim_Mat_LB = self._class_Sim_Mat_LB
        tr_data._class_Sim_Mat_UB = self._class_Sim_Mat_UB
        tr_data.CheckClassSpace()

        te_data = TableDataSet()
        te_data._x = self._x[~is_train]
        te_data._y = self._y[~is_train]
        te_data._class_Sim_Mat_LB = self._class_Sim_Mat_LB
        te_data._class_Sim_Mat_UB = self._class_Sim_Mat_UB
        te_data.CheckClassSpace()

        return {'train_data_set': tr_data, 'test_data_set': te_data}

    def To(self, device=torch.device('cpu')):
        self._x = torch.tensor(data=self._x, device=device)
        self._y = torch.tensor(data=self._y, device=device)
        self._class_Sim_Mat_UB = torch.tensor(data=self._class_Sim_Mat_UB, device=device)
        self._class_Sim_Mat_LB = torch.tensor(data=self._class_Sim_Mat_LB, device=device)


class ImageNet(MyDataSet):
    image_trans = transforms.Compose([transforms.Resize(size=[224, 224])])

    def __init__(self, data_dir, cla_inf: ClassSpaceInformation, train_val_test='train', val_file_class=None):
        """

        :param data_dir: string,
        (1) training __data set:
        There are many folders under this path,
        Each class corresponds to a folder,
        Each file corresponds to a sample (image)
        (2) val __data set:
        There are many files under this path,
        Each file corresponds to a sample (image),
        class label is in the file name
        (3) test __data set:
        There are many files under this path,
        Each file corresponds to a sample (image),
        No class label is given
        """
        super(MyDataSet, self).__init__()
        self._x = []
        self._y = []
        cla_id_list = cla_inf.GetClassID_list()

        if train_val_test == 'train':
            for cla_id in cla_id_list:
                cla_str = cla_inf.id2folder_name(class_id=cla_id)
                file_list = os.listdir(data_dir + cla_str + '/')
                for file_name in file_list:
                    self._x.append(data_dir + cla_str + '/' + file_name)
                    self._y.append(cla_id)
                    # print(data_dir + folder + '/' + file_name)
        elif train_val_test == 'val':
            f_reader = open(file=val_file_class, mode='r')
            line = f_reader.readline()
            while line != '':
                line = line.replace('\n', '')
                res = line.split('\t')
                file_name = res[0]
                cla_str = res[1]
                self._x.append(data_dir + file_name + '.JPEG')
                self._y.append(cla_inf.folder_name2id(class_folder_name=cla_str))
                line = f_reader.readline()
            f_reader.close()

        elif train_val_test == 'test':
            file_list = os.listdir(data_dir)
            for file_name in file_list:
                self._x.append(data_dir + file_name)
                self._y.append(ClassSpaceInformation.unknown_class_id)
        else:
            raise Exception('parameter train_val_test=' + str(train_val_test) + 'is not allowed.')
        self._y = np.array(self._y).astype(np.int64)

    def __preprocessing__(self, x: Tensor) -> Tensor:
        x = ImageNet.image_trans(x)
        if x.shape[0] == 1:
            x = torch.cat(tensors=[x, x, x], dim=0)
        return x


class Caltech256(MyDataSet):
    image_trans = transforms.Compose([transforms.Resize(size=[224, 224])])

    def __init__(self, data_dir, cla_inf: ClassSpaceInformation):
        """

        :param data_dir: string,
        There are many folders under this path,
        Each class corresponds to a folder,
        Each file corresponds to a sample (image)
        """
        super(MyDataSet, self).__init__()
        cla_space = cla_inf.GetClassID_list()
        self._x = []
        self._y = []
        for cla_id in cla_space:
            cls_str = cla_inf.id2folder_name(class_id=cla_id)
            file_list = os.listdir(data_dir + cls_str + '/')
            for file_name in file_list:
                self._x.append(data_dir + cls_str + '/' + file_name)
                self._y.append(cla_id)

        self._y = np.array(self._y).astype(np.int64)

    def __preprocessing__(self, x: Tensor) -> Tensor:
        x = Caltech256.image_trans(x)
        if x.shape[0] == 1:
            x = torch.cat(tensors=[x, x, x], dim=0)
        return x


class CIFAR10(MyDataSet):
    def __init__(self, mat_file_name):
        super(CIFAR10, self).__init__()
        Xy = hdf5storage.loadmat(file_name=mat_file_name)
        self._y = np.squeeze(Xy['y']).astype(np.int64)
        self._x = Xy['X'].astype(np.float64)
        image_channel = 3
        image_width = 32
        image_high = 32
        self._x = np.reshape(a=self._x, newshape=[self._y.shape[0],
                                                  image_channel,
                                                  image_width,
                                                  image_high])

    @staticmethod
    def LoadTrainDataset_FER():
        dataset = CIFAR10(mat_file_name='/root/cjbDisk/DataSet/CIFAR10/CIFAR10_train_50K.mat')
        res = hdf5storage.loadmat(file_name='/root/cjbDisk/DataSet/CIFAR10/CIFAR10_FER.mat')
        dataset._class_Sim_Mat_LB = res['LB']
        dataset._class_Sim_Mat_UB = res['UB']
        dataset.CheckClassSpace()
        return dataset

    @staticmethod
    def LoadTrainDataset_FER_5K():
        dataset = CIFAR10(mat_file_name='/root/cjbDisk/DataSet/CIFAR10/CIFAR10_train_5000.mat')
        res = hdf5storage.loadmat(file_name='/root/cjbDisk/DataSet/CIFAR10/CIFAR10_FER.mat')
        dataset._class_Sim_Mat_LB = res['LB']
        dataset._class_Sim_Mat_UB = res['UB']
        dataset.CheckClassSpace()
        return dataset

    @staticmethod
    def LoadTrainDataset_FER_500():
        dataset = CIFAR10(mat_file_name='/root/cjbDisk/DataSet/CIFAR10/CIFAR10_train_500.mat')
        res = hdf5storage.loadmat(file_name='/root/cjbDisk/DataSet/CIFAR10/CIFAR10_FER.mat')
        dataset._class_Sim_Mat_LB = res['LB']
        dataset._class_Sim_Mat_UB = res['UB']
        dataset.CheckClassSpace()
        return dataset

    @staticmethod
    def LoadTrainDataset_alpha_beta(alpha=0.2, beta=0.8):
        dataset = CIFAR10(mat_file_name='/root/cjbDisk/DataSet/CIFAR10/CIFAR10_train_50K.mat')
        dataset.SetClassFSR_LUB(alpha=alpha, beta=beta)
        dataset.CheckClassSpace()
        return dataset

    @staticmethod
    def LoadTestDataset():
        dataset = CIFAR10(mat_file_name='/root/cjbDisk/DataSet/CIFAR10/CIFAR10_test_10K.mat')
        dataset.CheckClassSpace()
        return dataset


class CIFAR100(MyDataSet):

    def __init__(self, mat_file_name, is_image=True):
        super(CIFAR100, self).__init__()
        Xy = hdf5storage.loadmat(file_name=mat_file_name)
        self._y = np.squeeze(Xy['y']).astype(np.int64)
        self._x = Xy['X'].astype(np.float64)
        if is_image:
            self._x = np.reshape(a=self._x, newshape=[self._y.shape[0], 3, 32, 32])

    @staticmethod
    def LoadTrainDataset_3Layer():
        data_set = CIFAR100(mat_file_name='/root/cjbDisk/DataSet/CIFAR100/CIFAR100_train_50K.mat')
        hie_par = HierarchicalPartition.FromTextFile(
            file_name='/root/cjbDisk/DataSet/CIFAR100/class_hierarchical_structure.txt')
        hie_par.lambda_list = [1e-3, 0.5, 1.0]
        fer = hie_par.ToFuzzyEquivalentRelation()
        LB, UB = fer.ToLowerUpperBound()
        data_set._class_Sim_Mat_LB = LB
        data_set._class_Sim_Mat_UB = UB
        data_set.CheckClassSpace()
        return data_set

    @staticmethod
    def LoadTrainDataset_WordNet():
        data_set = CIFAR100(mat_file_name='/root/cjbDisk/DataSet/CIFAR100/CIFAR100_train_50K.mat')
        fer_mat = hdf5storage.loadmat(file_name='/root/cjbDisk/DataSet/CIFAR100/CIFAR100_FER.mat')['fer']
        fer = FuzzyRelation(mat=fer_mat)
        if not FuzzyRelation.IsFuzzyEquivalentRelation(FR=fer):
            raise Warning('FuzzyRelation.IsFuzzyEquivalentRelation(FR=fer) == False!!')
        LB, UB = fer.ToLowerUpperBound()
        data_set._class_Sim_Mat_LB = LB
        data_set._class_Sim_Mat_UB = UB
        data_set.CheckClassSpace()
        return data_set

    @staticmethod
    def LoadTrainDataset_alpha_beta(alpha=0.2, beta=0.8):
        data_set = CIFAR100(mat_file_name='/root/cjbDisk/DataSet/CIFAR100/CIFAR100_train_50K.mat')
        data_set.SetClassFSR_LUB(alpha=alpha, beta=beta)
        data_set.CheckClassSpace()
        return data_set

    @staticmethod
    def LoadTestDataset():
        data_set = CIFAR100(mat_file_name='/root/cjbDisk/DataSet/CIFAR100/CIFAR100_test_10K.mat')
        data_set.CheckClassSpace()
        return data_set

    @staticmethod
    def LoadTrainDataset_Resnet101_alpha_beta(alpha=0.2, beta=0.8):
        data_set = CIFAR100(mat_file_name='/root/cjbDisk/DataSet/CIFAR100/CIFAR100_ResNet101_train_50K.mat',
                            is_image=False)
        data_set.SetClassFSR_LUB(alpha=alpha, beta=beta)
        data_set.CheckClassSpace()
        return data_set

    @staticmethod
    def LoadTrainDataset_Resnet101():
        data_set = CIFAR100(mat_file_name='/root/cjbDisk/DataSet/CIFAR100/CIFAR100_ResNet101_train_50K.mat',
                            is_image=False)
        fer_mat = hdf5storage.loadmat(file_name='/root/cjbDisk/DataSet/CIFAR100/CIFAR100_FER.mat')['fer']
        fer = FuzzyRelation(mat=fer_mat)
        if not FuzzyRelation.IsFuzzyEquivalentRelation(FR=fer):
            raise Warning('FuzzyRelation.IsFuzzyEquivalentRelation(FR=fer) == False!!')
        LB, UB = fer.ToLowerUpperBound()
        data_set._class_Sim_Mat_LB = LB
        data_set._class_Sim_Mat_UB = UB
        data_set.CheckClassSpace()
        return data_set

    @staticmethod
    def LoadTestDataset_Resnet101():
        data_set = CIFAR100(mat_file_name='/root/cjbDisk/DataSet/CIFAR100/CIFAR100_ResNet101_test_10K.mat',
                            is_image=False)
        data_set.CheckClassSpace()
        return data_set


class MNIST(MyDataSet):
    def __init__(self, mat_file_name, class_FER_UB_LB_mat_file=None):
        super(MNIST, self).__init__()
        Xy = hdf5storage.loadmat(file_name=mat_file_name)
        self._y = np.squeeze(Xy['y']).astype(np.int64)
        self._x = Xy['X'].astype(np.float64)
        image_channel = 1
        image_width = 28
        image_high = 28
        self._x = np.reshape(a=self._x, newshape=[self._y.shape[0],
                                                  image_channel,
                                                  image_width,
                                                  image_high])

        if class_FER_UB_LB_mat_file is not None:
            UL = hdf5storage.loadmat(file_name=class_FER_UB_LB_mat_file)
            self._class_Sim_Mat_UB = UL['FER']
            self._class_Sim_Mat_LB = UL['FER_LB']

    def To(self, device=torch.device('cpu')):
        self._x = torch.tensor(data=self._x, device=device)
        self._y = torch.tensor(data=self._y, device=device)
        if self._class_Sim_Mat_UB is not None:
            self._class_Sim_Mat_UB = torch.tensor(data=self._class_Sim_Mat_UB, device=device)
            self._class_Sim_Mat_LB = torch.tensor(data=self._class_Sim_Mat_LB, device=device)


    def Image_0_255_to_neg1_pos1(self):
        """
        for image __data,
        convert {0, 1, ... 255} to [-1, 1]
        :return:
                """
        self._x = self._x - 127.5
        self._x = self._x / 127.5

    def SortByClass(self):
        sort_ind = np.argsort(self._y)
        self._y = self._y[sort_ind]
        self._x = self._x[sort_ind]

    @staticmethod
    def LoadTrainDataset():
        data = MNIST(mat_file_name='/root/cjbDisk/DataSet/MNIST/MNIST_train_60K.mat',
                     class_FER_UB_LB_mat_file='/root/cjbDisk/DataSet/MNIST/10Annotation_class_FER_LB_UB.mat')
        data.CheckClassSpace()
        return data

    @staticmethod
    def LoadTrainDataset_alpha_beta(alpha=0.2, beta=0.8):
        data = MNIST(mat_file_name='/root/cjbDisk/DataSet/MNIST/MNIST_train_60K.mat')
        data.SetClassFSR_LUB(alpha=alpha, beta=beta)
        data.CheckClassSpace()
        return data

    @staticmethod
    def LoadTestDataset():
        data = MNIST(mat_file_name='/root/cjbDisk/DataSet/MNIST/MNIST_test_10K.mat')
        data.CheckClassSpace()
        return data


class PASCAL_VOC(MyDataSet):
    def __init__(self, data_dir, image_class_file):
        super(PASCAL_VOC, self).__init__()
        self._x = []
        self._y = []
        f = open(file=image_class_file, mode='r')
        line = f.readline()
        while line != '':
            items = line.split('\t')
            self._x.append(data_dir + items[0])
            self._y.append(np.int64(items[1]))
            line = f.readline()
        f.close()
        self._y = np.array(self._y).astype(np.int64)

    @staticmethod
    def Load2007train_data():
        data = PASCAL_VOC(data_dir='/root/cjbDisk/DataSet/PASCAL-VOC/2007train/JPEGImages/',
                          image_class_file='/root/cjbDisk/DataSet/PASCAL-VOC/2007train_image_class.txt')
        data.CheckClassSpace()
        return data

    @staticmethod
    def Load2007test_data():
        data = PASCAL_VOC(data_dir='/root/cjbDisk/DataSet/PASCAL-VOC/2007test/JPEGImages/',
                          image_class_file='/root/cjbDisk/DataSet/PASCAL-VOC/2007test_image_class.txt')
        data.CheckClassSpace()
        return data


class ImagePreprocessing(nn.Module):
    def __init__(self, prob_processing_list=None):
        """

        :param prob_processing_list: prob_processing_list[i]: {'prob': 0.3, 'processing': nn.Module}
        """
        super(ImagePreprocessing, self).__init__()
        self.__prob_processing_list = prob_processing_list

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: shape=[num_sam, num_channel, high, width]
        :return:
        """
        with torch.no_grad():
            if self.__prob_processing_list is None:
                return x
            else:
                n_sam = x.shape[0]
                for i in range(n_sam):
                    x[i] = self.__processing_one_sample(x=x[i])
                return x

    def __processing_one_sample(self, x: Tensor) -> Tensor:
        """

        :param x: shape=[num_channel, high, width]
        :return:
        """
        n = self.__prob_processing_list.__len__()
        prob = torch.rand(size=[n])
        for i in range(n):
            prob_processing = self.__prob_processing_list[i]
            if prob[i] <= prob_processing['prob']:
                x = prob_processing['processing'](x)
        return x

    def ToStr(self):
        rep_str = str(type(self)) + '=\n'
        if self.__prob_processing_list is None:
            rep_str += 'None'
        else:
            for prob_processing in self.__prob_processing_list:
                rep_str += 'prob=\t' + str(prob_processing['prob']) + \
                           '\tprocessing=\t' + prob_processing['processing'].__repr__() + '\n'
        return rep_str

    @staticmethod
    def MNIST_Preprocessing(prob, device=torch.device('cpu')):
        m = ImagePreprocessing(prob_processing_list=[
            {'prob': prob, 'processing': transforms.RandomAffine(degrees=(-20, 20), translate=(0.2, 0.2))}
        ])
        m.to(device)
        return m

    @staticmethod
    def CIFAR100_Preprocessing(device=torch.device('cpu')):
        m = ImagePreprocessing(prob_processing_list=[
            {'prob': 1, 'processing': transforms.RandomHorizontalFlip()},
            {'prob': 1, 'processing': transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))}
            # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ])
        m.to(device)
        return m

    @staticmethod
    def CIFAR100_Preprocessing_test(device=torch.device('cpu')):
        m = ImagePreprocessing(prob_processing_list=[
            {'prob': 1, 'processing': transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))}
        ])
        m.to(device)
        return m

    @staticmethod
    def CIFAR10_Preprocessing(device=torch.device('cpu')):
        m = ImagePreprocessing(prob_processing_list=[
            {'prob': 1, 'processing': transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))},
            {'prob': 1, 'processing': transforms.RandomCrop(32, padding=4)},
            {'prob': 1, 'processing': transforms.RandomHorizontalFlip()},
        ])
        m.to(device)
        return m

    @staticmethod
    def CIFAR10_Preprocessing_test(device=torch.device('cpu')):
        m = ImagePreprocessing(prob_processing_list=[
            {'prob': 1, 'processing': transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}
        ])
        m.to(device)
        return m


class BatchGenerator(object):
    def __init__(self):
        super(BatchGenerator, self).__init__()

    def InitEpoch(self, batch_size, batch_class_num=None, epoch_num=1, do_shuffle=True):
        pass

    def NextBatch(self):
        pass


class Partition_BG(BatchGenerator):
    def __init__(self, data_set: MyDataSet):
        super(Partition_BG, self).__init__()
        self._data_set = data_set
        self._batch_size = None
        self._next_batch_beg_id = None
        self._sam_id_arr = None

    def InitEpoch(self, batch_size, batch_class_num=None, epoch_num=1, do_shuffle=True):
        self._batch_size = batch_size
        self._next_batch_beg_id = 0
        sam_id_arr = list(range(self._data_set.GetSampleNum())) * epoch_num
        sam_id_arr = np.array(sam_id_arr)
        if do_shuffle:
            np.random.shuffle(sam_id_arr)
        self._sam_id_arr = sam_id_arr

    def NextBatch(self):
        bge_i = self._next_batch_beg_id
        end_i = bge_i + self._batch_size
        is_last_batch = False
        if end_i >= self._sam_id_arr.shape[0]:
            end_i = self._sam_id_arr.shape[0]
            is_last_batch = True

        sam_indices = self._sam_id_arr[bge_i:end_i]
        res = self._data_set.GetBatchX_y(sam_indices=sam_indices)
        self._next_batch_beg_id = end_i

        return {'X': res['X'], 'y': res['y'], 'is_last_batch': is_last_batch}


class ClassSample_BG(BatchGenerator):
    def __init__(self, data_set: MyDataSet):
        super(BatchGenerator, self).__init__()
        self._data_set = data_set
        self._class_space = data_set.GetClassSpace()
        self._sam_num = data_set.GetSampleNum()
        self._class_sam_ind_map = {}
        ind_bank = np.arange(start=0, stop=self._sam_num)
        y = data_set.GerAll_y()
        for c in self._class_space:
            Ic = y == c
            self._class_sam_ind_map[c] = ind_bank[Ic]

        self._batch_size = None
        self._batch_class_size = None
        self._batch_class_num = None
        self._rest_sam_num = None

    def InitEpoch(self, batch_size, batch_class_num=None, epoch_num=1, do_shuffle=True):
        self._batch_class_num = batch_class_num
        if batch_class_num > self._class_space.shape[0]:
            self._batch_class_num = self._class_space.shape[0]

        self._batch_size = batch_size
        self._batch_class_size = batch_size // self._batch_class_num
        if batch_size % batch_class_num != 0:
            self._batch_class_size += 1

        self._rest_sam_num = self._sam_num

    def NextBatch(self):
        """
        random class subset: {c1, c2, ..., cm}
        random sample subset: {(x1, c1), (x2,c1), ..., (x_n, c1)}
                            U {(x1, c2), (x2,c2), ..., (x_n, c2)}
                            U ...
                            U {(x1, cm), (x2,cm), ..., (x_n, cm)}
        :return:
        """
        class_arr = np.random.choice(a=self._class_space, size=self._batch_class_num, replace=False)
        sam_ind = []
        for c in class_arr:
            ind_c = self._class_sam_ind_map[c]
            replace = ind_c.shape[0] < self._batch_class_size
            ind_c = np.random.choice(a=ind_c, size=self._batch_class_size, replace=replace)
            sam_ind.append(ind_c)
        sam_ind = np.concatenate(sam_ind, axis=0)
        self._rest_sam_num -= sam_ind.shape[0]
        res = self._data_set.GetBatchX_y(sam_indices=sam_ind)
        return {'X': res['X'], 'y': res['y'], 'is_last_batch': self._rest_sam_num <= 0}
