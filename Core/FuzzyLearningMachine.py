import numpy as np
import torch
from torch import nn, optim
from copy import copy
from cvxopt import matrix, solvers

from Core.FeatureExtractionNetwork import FeatureExtractionNetwork
from Core.FuzzySimilarityRelationNetwork import FuzzySimilarityRelationNetwork
from Core.FuzzinessPermissibleLoss import FuzzinessPermissibleLoss
from Core.IntervalRegressionLoss import IntervalRegressionLoss
from Core.SimilarityRankLoss import SimilarityRankLoss
from Core.RegularizationTerm import RegularizationTerm
from Core.DataSet import MyDataSet, ImagePreprocessing
from Core.Kit import Tool

"""
fea: Feature
ext: Extraction
net: Network
num: Number
LB: Lower Bound
UB: Upper Bound
Mat: Matrix
"""


class FuzzyLearningMachine(nn.Module):
    def __init__(self, fea_ext_net: FeatureExtractionNetwork, fsr_net: FuzzySimilarityRelationNetwork,
                 FPL_alpah=0.2, FPL_beta=0.8, gamma_FPL=1.0, gamma_int_reg=1.0, gamma_sim_rank=1.0, gamma_reg=0.0):
        """

        :param fea_ext_net: Feature Extraction Network
        :param fsr_net: Fuzzy Similarity Relation Network
        :param FPL_alpah: Parameter of Fuzziness Permissible Loss
        :param FPL_beta: Parameter of Fuzziness Permissible Loss
        :param gamma_FPL: the weight of Fuzziness Permissible Loss
        :param gamma_int_reg: the weight of Similarity Interval Regression Loss
        :param gamma_sim_rank: the weight of Similarity Rank Loss
        :param gamma_reg: the weight of Regularization (default: (2-norm)^2)
        """
        super(FuzzyLearningMachine, self).__init__()
        self.__fea_ext_net = fea_ext_net
        self.__fsr_net = fsr_net

        self.__gamma_FPL = gamma_FPL
        self.__gamma_int_reg = gamma_int_reg
        self.__gamma_smi_rank = gamma_sim_rank
        self.__gamma_reg = gamma_reg

        self.__fuzz_per_loss = FuzzinessPermissibleLoss(alpha=FPL_alpah, beta=FPL_beta)
        self.__int_reg_loss = IntervalRegressionLoss()
        self.__smi_rank_loss = SimilarityRankLoss()
        self.__reg_term = RegularizationTerm()

        self.__optimizer = None

    def ToStr(self):
        rep_str = self.__repr__()
        rep_str += '\n'
        rep_str += 'gamma_FPL=\t' + str(self.__gamma_FPL) + '\n'
        rep_str += 'gamma_int_reg=\t' + str(self.__gamma_int_reg) + '\n'
        rep_str += 'gamma_smi_rank=\t' + str(self.__gamma_smi_rank) + '\n'
        rep_str += 'gamma_reg=\t' + str(self.__gamma_reg) + '\n'
        rep_str += 'Para-ID\tname\ttype\tshape\tnumel\n'
        paras = self.named_parameters()
        i = 0
        para_num = 0
        for name, para in paras:
            rep_str += str(i) + '\t' + name + '\t' + str(type(para)) + '\t' \
                       + str(para.shape) + '\t' + str(para.numel()) + '\n'
            para_num += para.numel()
            i += 1
        rep_str += 'number of parameters=\t' + str(para_num) + '\n'
        return rep_str

    def SaveSetting2File(self, file_name):
        f = open(file=file_name, mode='w')
        f.write(self.ToStr())
        f.close()

    def __Update_parameters(self, batch_data, preprocessing: ImagePreprocessing = None, device=torch.device('cpu'),
                            times=10):
        y = batch_data['y'].to(device)  # shape=[batch_size]
        X = batch_data['X'].to(device)  # shape=[batch_size, sample.shape]
        if preprocessing is not None:
            X = preprocessing(X)

        fuzz_per_loss = -np.inf
        int_reg_loss = -np.inf
        sim_rank_loss = -np.inf
        reg = -np.inf
        for _ in range(times):
            self.__optimizer.zero_grad()
            H = self.__fea_ext_net.ComputeLatentRepresentation(X=X)  # shape=[batch_size, latent_fea_num]
            S = self.__fsr_net.ComputeFSR_Mat(H=H)  # shape=[batch_size, batch_size]

            total = None
            if self.__gamma_FPL > 0.0:
                fuzz_per_loss = self.__fuzz_per_loss.ComputeLoss(FSR_Mat=S, labels=y)
                total = self.__gamma_FPL * fuzz_per_loss
                fuzz_per_loss = fuzz_per_loss.detach().clone().cpu().numpy()

            if self.__gamma_int_reg > 0.0:
                int_reg_loss = self.__int_reg_loss.ComputeLoss(FSR_Mat=S, labels=y)
                total = self.__gamma_int_reg * int_reg_loss
                int_reg_loss = int_reg_loss.detach().clone().cpu().numpy()

            if self.__gamma_smi_rank > 0.0:
                sim_rank_loss = self.__smi_rank_loss.ComputeLoss(FSR_Mat=S, labels=y)
                total += self.__gamma_smi_rank * sim_rank_loss
                sim_rank_loss = sim_rank_loss.detach().clone().cpu().numpy()

            if self.__gamma_reg > 0.0:
                reg = self.__reg_term(m=self)
                total += self.__gamma_reg * reg

                reg = reg.detach().clone().cpu().numpy()

            total.backward()
            self.__optimizer.step()

        return {'fuzz_per_loss': fuzz_per_loss, 'int_reg_loss': int_reg_loss,
                'smi_rank_loss': sim_rank_loss, 'reg': reg}

    def Train_with_log(self, train_data_set: MyDataSet, preprocessing: ImagePreprocessing = None, max_epoch_num=100,
                       batch_size=64, batch_class_num=None, lr=1e-3, how_often_decay_lr=10000000, decay_factor=1.0,
                       test_preprocessing: ImagePreprocessing = None, test_data_set: MyDataSet = None,
                       class_exemplar_num=5, exemplar_type='density', train_log_prefix=None,
                       device=torch.device('cpu')):
        """

        :param train_data_set:
        :param preprocessing:
        :param max_epoch_num:
        :param batch_size:
        :param batch_class_num: 
        :param lr:
        :param how_often_decay_lr:
        :param decay_factor:
        :param test_preprocessing:
        :param test_data_set:
        :param class_exemplar_num:
        :param exemplar_type: {'density', 'density-peek'}:
        :param train_log_prefix:
        :param device:
        :return:
        """

        self.train()
        LUB = train_data_set.GetClassSmi_LUB()
        self.__int_reg_loss.SetClassSim_LUB(class_smi_UB=LUB['UB'].to(device), class_smi_LB=LUB['LB'].to(device))
        class_sim_mat = train_data_set.GetClassSimilarityMat()
        self.__smi_rank_loss.SetClassSimMat(class_sim_mat=class_sim_mat.to(device))
        self.__optimizer = optim.Adam(params=self.parameters(), lr=lr)


        log = None
        if train_log_prefix is not None:
            log = open(file=train_log_prefix + '.txt', mode='w')
            log.write('max_epoch_num=\t' + str(max_epoch_num) + '\n' +
                      'batch_size=\t' + str(batch_size) + '\n' +
                      'lr=\t' + str(lr) + '\n' +
                      'how_often_decay_lr=\t' + str(how_often_decay_lr) + '\n' +
                      'decay_factor=\t' + str(decay_factor) + '\n')
            log.write('ImagePreprocessing=\t')
            if preprocessing is None:
                log.write('None\n')
            else:
                log.write(preprocessing.ToStr() + '\n')
            log.write('test-ImagePreprocessing=\t')
            if test_preprocessing is None:
                log.write('None\n')
            else:
                log.write(test_preprocessing.ToStr() + '\n')
            log.flush()

        best_acc = -1
        train_data_set.ResetBatchGenerator()
        for epo_ind in range(1, max_epoch_num + 1):
            fuzz_per_loss_list = []
            int_reg_loss_list = []
            smi_rank_loss_list = []
            reg_list = []
            train_data_set.InitEpoch(batch_size=batch_size, batch_class_num=batch_class_num)
            while True:
                batch_data = train_data_set.NextBatch()
                res = self.__Update_parameters(batch_data=batch_data, preprocessing=preprocessing, device=device)

                fuzz_per_loss_list.append(res['fuzz_per_loss'])
                int_reg_loss_list.append(res['int_reg_loss'])
                smi_rank_loss_list.append(res['smi_rank_loss'])
                reg_list.append(res['reg'])

                if batch_data['is_last_batch']:
                    break

            fuzz_per_loss = np.mean(np.array(fuzz_per_loss_list))
            int_reg_loss = np.mean(np.array(int_reg_loss_list))
            smi_rank_loss = np.mean(np.array(smi_rank_loss_list))
            reg = np.mean(np.array(reg_list))

            acc = -1
            if test_data_set is not None and epo_ind % 1 == 0:
                res = self.SelectExemplarPredict(train_data_set=train_data_set, test_dataset=test_data_set,
                                                 preprocessing=test_preprocessing, device=device,
                                                 class_exemplar_num=class_exemplar_num, exemplar_type=exemplar_type)
                acc = res['acc']
                if acc > best_acc:
                    best_acc = acc
                    if train_log_prefix is not None:
                        self.Save2File(file_name=train_log_prefix + 'BestModel.model')
                        Tool.SavePredictResult(res=res, file_name=train_log_prefix + 'BestPredictResult.txt')

            print(Tool.TimeStr() + '\tepoch=\t' + str(epo_ind) + '\tend,' +
                  '\tfuzz_per_loss=\t' + str(fuzz_per_loss) +
                  '\tint_reg_loss=\t' + str(int_reg_loss) +
                  '\tsmi_rank_loss=\t' + str(smi_rank_loss) +
                  '\treg=\t' + str(reg) +
                  '\texemplar-test-acc=\t' + str(acc))
            if log is not None:
                log.write(Tool.TimeStr() + '\tepoch=\t' + str(epo_ind) + '\tend,' +
                          '\tfuzz_per_loss=\t' + str(fuzz_per_loss) +
                          '\tint_reg_loss=\t' + str(int_reg_loss) +
                          '\tsmi_rank_loss=\t' + str(smi_rank_loss) +
                          '\treg=\t' + str(reg) +
                          '\texemplar-test-acc=\t' + str(acc) + '\n')
                log.flush()

            if epo_ind % how_often_decay_lr == 0:
                self.decay_learning_rate(decay_factor=decay_factor)
                if log is not None:
                    log.write('\nself.decay_learning_rate(decay_factor=' + str(decay_factor) + ')\n')
                    log.flush()

        if log is not None:
            log.write('Train-end.')
            log.close()

    def decay_learning_rate(self, decay_factor=0.9):
        """
        :param decay_factor:
        :return:
        """
        params = self.__optimizer.param_groups
        for i in range(len(params)):
            params[i]['lr'] *= decay_factor

    def set_learning_rate(self, lr=1e-3):
        """
        :param lr:
        :return:
        """
        params = self.__optimizer.param_groups
        for i in range(len(params)):
            params[i]['lr'] = lr

    def SetGamma(self, gamma_FPL, gamma_int_reg, gamma_sim_rank, gamma_reg):
        self.__gamma_FPL = gamma_FPL
        self.__gamma_int_reg = gamma_int_reg
        self.__gamma_smi_rank = gamma_sim_rank
        self.__gamma_reg = gamma_reg

    def PredictFSR_Mat(self, data_set1: MyDataSet, data_set2: MyDataSet, preprocessing: ImagePreprocessing = None,
                       batch_size=10000, device=torch.device('cpu')) -> np.array:
        is_train_state = self.training
        self.eval()
        row_list = []
        data_set1.InitEpoch(batch_size=batch_size, do_shuffle=False)
        while True:
            batch_data1 = data_set1.NextBatch()
            r_X = batch_data1['X'].to(device)
            if preprocessing is not None:
                r_X = preprocessing(r_X)
            r_H = self.__fea_ext_net.ComputeLatentRepresentation(X=r_X)
            # ####### data_set2 begin ##############
            block_list = []
            data_set2.InitEpoch(batch_size=batch_size, do_shuffle=False)
            while True:
                batch_data2 = data_set2.NextBatch()
                c_X = batch_data2['X'].to(device)
                if preprocessing is not None:
                    c_X = preprocessing(c_X)
                c_H = self.__fea_ext_net.ComputeLatentRepresentation(X=c_X)
                rc_S = self.__fsr_net.ComputeFSR_Mat2(H1=r_H, H2=c_H)
                block_list.append(rc_S.detach().clone().cpu().numpy())
                if batch_data2['is_last_batch']:
                    break
            # ####### data_set2 end ##############
            row_list.append(np.concatenate(block_list, axis=1))
            if batch_data1['is_last_batch']:
                break
        if is_train_state:
            self.train()
        return np.concatenate(row_list, axis=0)

    def PredictClass_FSR(self, data_set: MyDataSet, batch_size=10000, device=torch.device('cpu')):
        """

        :param data_set: class space must be {0, 1, 2, ..., num_class-1}
        :param batch_size:
        :param device:
        :return:
        """
        is_train = self.training
        self.eval()
        sam_num = data_set.GetSampleNum()
        class_num = data_set.GetClassNum()
        S = np.zeros(shape=[class_num, class_num])
        cla_sam_num, sam_ind = data_set.GetClassSampleNum()
        with torch.no_grad():
            beg1 = 0
            while beg1 < sam_num:
                end1 = beg1 + batch_size
                if end1 > sam_num:
                    end1 = sam_num
                B1 = data_set.GetBatchX_y(sam_indices=sam_ind[beg1: end1])
                X1 = B1['X'].to(device)
                y1 = B1['y'].to(device)
                class_space1 = torch.unique(y1)
                H1 = self.__fea_ext_net.ComputeLatentRepresentation(X=X1)
                #  ========== Inner-beg ============
                beg2 = 0
                while beg2 < sam_num:
                    end2 = beg2 + batch_size
                    if end2 > sam_num:
                        end2 = sam_num
                    B2 = data_set.GetBatchX_y(sam_indices=sam_ind[beg2: end2])
                    X2 = B2['X'].to(device)
                    y2 = B2['y'].to(device)
                    class_space2 = torch.unique(y2)
                    H2 = self.__fea_ext_net.ComputeLatentRepresentation(X=X2)
                    S12 = self.__fsr_net.ComputeFSR_Mat2(H1=H1, H2=H2)  # shape=[batch_size, batch_size]

                    for c1 in class_space1:
                        temp = S12[y1 == c1, :]
                        for c2 in class_space2:
                            S[c1, c2] += torch.sum(temp[:, y2 == c2]).detach().cpu().numpy()

                    beg2 = end2
                #  ========== Inner-end ============
                beg1 = end1
        D = np.diag(1.0 / cla_sam_num)
        S = np.matmul(D, S)
        S = np.matmul(S, D)
        diag_I = np.arange(0, class_num)
        S[diag_I, diag_I] = 1.0

        if is_train:
            self.train()
        return S

    def PredictClass_FSR1(self, data_set: MyDataSet, batch_size=10000, device=torch.device('cpu')):
        """

        :param data_set: class space must be {0, 1, 2, ..., num_class-1}
        :param batch_size:
        :param device:
        :return:
        """
        sam_FSR = self.PredictFSR_Mat(data_set1=data_set, data_set2=copy(data_set), batch_size=batch_size,
                                      device=device)
        num_class = data_set.GetClassNum()
        y = data_set.GerAll_y()
        class_FSR = - np.inf * np.ones(shape=[num_class, num_class])
        for i in range(num_class):
            class_FSR[i, i] = 1.0
            Ii = y == i
            Si = sam_FSR[Ii, :]
            for j in range(i + 1, num_class):
                Ij = y == j
                Sij = Si[:, Ij]
                s = np.mean(Sij)
                class_FSR[i, j] = s
                class_FSR[j, i] = s
        return class_FSR

    def PredictClass_FSR2(self, data_set: MyDataSet, device=torch.device('cpu')):
        """

        :param data_set: class space must be {0, 1, 2, ..., num_class-1}
        :param device:
        :return:
        """
        is_train_state = self.training
        self.eval()
        with torch.no_grad():
            num_class = data_set.GetClassNum()
            X, y = data_set.GetAll_x_y()
            class_FSR = - np.inf * np.ones(shape=[num_class, num_class])
            for i in range(num_class):
                class_FSR[i, i] = 1.0
                Xi = torch.tensor(data=X[y == i], device=device)
                Hi = self.__fea_ext_net.ComputeLatentRepresentation(X=Xi)
                for j in range(i + 1, num_class):
                    Xj = torch.tensor(data=X[y == j], device=device)
                    Hj = self.__fea_ext_net.ComputeLatentRepresentation(X=Xj)
                    Sij = self.__fsr_net.ComputeFSR_Mat2(H1=Hi, H2=Hj)
                    s = torch.mean(Sij).detach().clone().cpu().numpy()
                    class_FSR[i, j] = s
                    class_FSR[j, i] = s

        if is_train_state:
            self.train()
        return class_FSR

    def PredictClass_FSR3(self, data_set: MyDataSet, batch_size=10000, device=torch.device('cpu')):
        y = data_set.GerAll_y()
        num_class = np.unique(y).shape[0]
        class_FSR = np.zeros(shape=[num_class, num_class])
        row_ind, col_ind, class_sam_num = FuzzyLearningMachine.GenerateSamplePair(y=y)
        pair_num = row_ind.shape[0]
        is_train_state = self.training
        self.eval()
        begI = 0
        with torch.no_grad():
            while begI < pair_num:
                endI = begI + batch_size
                if endI > pair_num:
                    endI = pair_num
                B1 = data_set.GetBatchX_y(sam_indices=row_ind[begI:endI])
                X1 = torch.tensor(data=B1['X'], device=device)
                y1 = B1['y'].detach().cpu().numpy()

                B2 = data_set.GetBatchX_y(sam_indices=col_ind[begI:endI])
                X2 = torch.tensor(data=B2['X'], device=device)
                y2 = B2['y'].detach().cpu().numpy()

                H1 = self.__fea_ext_net.ComputeLatentRepresentation(X=X1)
                H2 = self.__fea_ext_net.ComputeLatentRepresentation(X=X2)
                s12 = self.__fsr_net.ComputeFSR_Mat2_diag(H1=H1, H2=H2)
                s12 = s12.detach().cpu().numpy()
                class_FSR[y1, y2] += s12
                begI = endI

        D = np.diag(1.0 / class_sam_num)

        class_FSR = np.matmul(D, class_FSR)
        class_FSR = np.matmul(class_FSR, D)
        class_FSR = class_FSR + np.eye(num_class)

        if is_train_state:
            self.train()
        return class_FSR

    @staticmethod
    def GenerateSamplePair(y: np.array):
        """

        :param y: shape=[num_sam]
        :return:
        """
        class_space = np.unique(y)
        class_num = class_space.shape[0]
        ind_back = np.arange(0, y.shape[0])
        row_ind = []
        col_ind = []
        class_sam_num = []
        for i in range(class_num):
            c1_ind = ind_back[y == class_space[i]]
            class_sam_num.append(c1_ind.shape[0])
            for j in range(i + 1, class_num):
                c2_ind = ind_back[y == class_space[j]]
                c12_ind = np.meshgrid(c1_ind, c2_ind)
                row_ind.append(np.reshape(a=c12_ind[0], newshape=[-1]))
                col_ind.append(np.reshape(a=c12_ind[1], newshape=[-1]))
        row_ind = np.concatenate(row_ind, axis=0)
        col_ind = np.concatenate(col_ind, axis=0)
        return row_ind, col_ind, np.array(class_sam_num)

    def SelectExemplar(self, data_set: MyDataSet, class_exemplar_num=5, batch_size=10000, exemplar_type='density',
                       preprocessing: ImagePreprocessing = None, device=torch.device('cpu')):
        """

        :param data_set:
        :param class_exemplar_num:
        :param batch_size:
        :param exemplar_type: {'density', 'density-peek'}
        :param preprocessing:
        :param device:
        :return:
        """

        is_train_state = self.training
        self.eval()
        with torch.no_grad():
            class_space = data_set.GetClassSpace()
            exemplar_dataset_list = []
            for c in class_space:
                # print('c=' + str(c))
                c_data = data_set.GetSubsetOfClass(class_id=c)
                # shape=[c_sam_num, c_sam_num]
                FSR_Mat_c = self.PredictFSR_Mat(data_set1=c_data, data_set2=copy(c_data), preprocessing=preprocessing,
                                                batch_size=batch_size, device=device)
                if exemplar_type == 'density':
                    sam_score = np.sum(FSR_Mat_c, axis=1)  # shape=[c_sam_num]
                elif exemplar_type == 'density-peek':
                    sam_score = FuzzyLearningMachine.DensityPeekSCore(sam_FSR=FSR_Mat_c)
                else:
                    raise Exception('exemplar_type=' + str(exemplar_type) + ' is Not Supported.')

                if class_exemplar_num > sam_score.shape[0]:
                    class_exemplar_num = sam_score.shape[0]

                sort_ind = np.argsort(-sam_score)  # ascending order
                exemplar_ind = sort_ind[0: class_exemplar_num]
                c_exemplar_data = c_data.GetSubset(sam_indices=exemplar_ind)
                # print(exemplar_ind)
                exemplar_dataset_list.append(c_exemplar_data)
        if is_train_state:
            self.train()
        return MyDataSet.UnionDataset(dataset_list=exemplar_dataset_list)

    @staticmethod
    def DensityPeekSCore(sam_FSR):
        """
        Rodriguez, Alex, and Alessandro Laio. "Clustering by fast search and find of density peaks."
        Science 344.6191 (2014): 1492-1496.
        :param sam_FSR: shape=[n_sam, n_sam], sam_FSR[i,i] = 1.0
        :return:
        """
        n_sam = sam_FSR.shape[0]
        density_arr = np.sum(sam_FSR, axis=1) / n_sam
        relatively_dis_arr = np.zeros(shape=[n_sam])
        for i in range(n_sam):
            Ii = density_arr > density_arr[i]
            if np.sum(Ii.astype(np.int32)) > 0:
                dis_arr = 1 - sam_FSR[i, Ii]
                relatively_dis_arr[i] = np.min(dis_arr)
            else:
                dis_arr = 1 - sam_FSR[i, :]
                relatively_dis_arr[i] = np.max(dis_arr)

        return density_arr * relatively_dis_arr

    def Predict_Exemplar(self, exemplar_dataset: MyDataSet, test_dataset: MyDataSet,
                         preprocessing: ImagePreprocessing = None, batch_size=10000, device=torch.device('cpu')):
        # shape=[exe_sam_num, te_sam_num]
        sam_exe_sim_Mat = self.PredictFSR_Mat(data_set1=exemplar_dataset, data_set2=test_dataset,
                                              preprocessing=preprocessing, batch_size=batch_size, device=device)
        sam_exe_sim_Mat = np.transpose(sam_exe_sim_Mat)  # shape=[te_sam_num, exe_sam_num]
        exe_y = exemplar_dataset.GerAll_y()
        class_space = exemplar_dataset.GetClassSpace()
        sam_cla_score_list = []
        for c in class_space:
            c_ind = exe_y == c
            sam_c_sim_Mat = sam_exe_sim_Mat[:, c_ind]  # shape=[te_sam_num, c_exe_num]
            sam_c_score = np.mean(sam_c_sim_Mat, axis=1, keepdims=True)  # shape=[te_sam_num, 1]
            sam_cla_score_list.append(sam_c_score)

        sam_class_score = np.concatenate(sam_cla_score_list, axis=1)  # shape=[te_sam_num, class_num]
        pre_ind = np.argmax(sam_class_score, axis=1)
        pre_y = class_space[pre_ind]
        te_y = test_dataset.GerAll_y()
        acc = np.mean((pre_y == te_y) + 0.0)
        return {'acc': acc, 'pre_y': pre_y, 'true_y': te_y}

    def SelectExemplarPredict(self, train_data_set: MyDataSet, test_dataset: MyDataSet,
                              class_exemplar_num=5, exemplar_type='density', batch_size=10000,
                              preprocessing: ImagePreprocessing = None, device=torch.device('cpu')):
        """

        :param train_data_set:
        :param test_dataset:
        :param class_exemplar_num:
        :param exemplar_type: {'density', 'density-peek'}
        :param batch_size:
        :param preprocessing:
        :param device:
        :return:
        """
        exe_data = self.SelectExemplar(data_set=train_data_set, class_exemplar_num=class_exemplar_num,
                                       exemplar_type=exemplar_type, batch_size=batch_size, preprocessing=preprocessing,
                                       device=device)
        # print('self.SelectExemplar() end.')
        res = self.Predict_Exemplar(exemplar_dataset=exe_data, test_dataset=test_dataset, batch_size=batch_size,
                                    preprocessing=preprocessing, device=device)
        # print('self.Predict_Exemplar() end.')
        return res

    def Save2File(self, file_name):
        torch.save(obj=self, f=file_name)

    @staticmethod
    def LoadFromFile(file_name, map_location=None):
        model = torch.load(f=file_name, map_location=map_location)
        return model

    @staticmethod
    def ComputeClassSimilarity(class_sim_Rank: np.array, class_sim_S: np.array, alpha=0.2, margin=1e-3):
        """

        :param class_sim_Rank: shape=[n_class, n_class], the class-similarity rank Derived from the class semantics
               R=R.T, R[i,i]=max(R)=m, R[i, j] in {0, 1, 2, ..., m},
               e.g., R[i,j]=2 > R[k,l]=1 <=> the sim(c_i, c_j) > sim(c_k, c_l) + margin
        :param class_sim_S: shape=[n_class, n_class], the class-similarity Derived from the samples
               0 <= class_sim_S[i, j] <= alpha, i,j = 1, 2, ..., n_class-1, i != j
               class_sim_S[i, j] = 1.0,  i = 1, 2, ..., n_class-1
        :param alpha: the upper bound of class-similarity between different classes
        :param margin: the margin of rank
        :return:
        """
        n_class = class_sim_S.shape[0]
        rank_space = []
        for i in range(n_class):  # skip the diag element
            for j in range(i + 1, n_class):
                rank_space.append(class_sim_Rank[i, j])
        rank_space = np.unique(np.array(rank_space))  # ascending order
        num_rank = rank_space.shape[0]
        s_arr = []
        for r in rank_space:
            ind = class_sim_Rank == r
            s = [np.max(class_sim_S[ind])]
            s_arr.append(s)
        s_arr = np.array(s_arr)
        """
        Assign a similarity value to each ranking
        min_{x} ||x-s_arr||_2^2
          s.t.  0<= x_i <= alpha, i = 1, 2, ..., num_rank
                x_{i+1} >= x_i + margin <=> x_i - x_{i+1} <= -margin, i=1, 2, ..., num_rank-1
        """
        P = matrix(2 * np.eye(num_rank))
        q = matrix(- 2 * s_arr)
        G1 = - np.eye(num_rank)
        h1 = np.zeros(shape=[num_rank, 1])
        G2 = np.eye(num_rank)
        h2 = alpha * np.ones(shape=[num_rank, 1])
        G3 = np.zeros(shape=[num_rank - 1, num_rank])
        h3 = - margin * np.ones(shape=[num_rank - 1, 1])
        for i in range(num_rank - 1):
            G3[i, i] = 1.0
            G3[i, i + 1] = - 1.0
        G = matrix(np.concatenate((G1, G2, G3), axis=0))
        h = matrix(np.concatenate((h1, h2, h3), axis=0))

        # print('P=')
        # print(P)
        # print('q=')
        # print(q)
        # print('G=')
        # print(G)
        # print('h=')
        # print(h)

        solvers.options['show_progress'] = False
        res = solvers.qp(P=P, q=q, G=G, h=h)
        solvers.options['show_progress'] = True
        x_star = np.array(res['x'])
        T = np.eye(n_class)
        for r in range(num_rank):
            ind = class_sim_Rank == rank_space[r]
            T[ind] = x_star[r]
        return {'T': T, 'loss': res['primal objective'] + np.sum(s_arr * s_arr)}
