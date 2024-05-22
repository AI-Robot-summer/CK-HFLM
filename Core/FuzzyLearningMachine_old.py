import numpy as np
import torch
from torch import nn, optim
from copy import copy

from Core.FeatureExtractionNetwork import FeatureExtractionNetwork
from Core.FuzzySimilarityRelationNetwork import FuzzySimilarityRelationNetwork
from Core.IntervalRegressionLoss import IntervalRegressionLoss
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
                 gamma_ce=0.0, gamma_int_reg=1.0, gamma_reg=0.0):
        """

        :param fea_ext_net: Feature Extraction Network
        :param fsr_net: Fuzzy Similarity Relation Network
        :param gamma_ce: the weight of Cross Entropy Loss
        :param gamma_int_reg: the weight of Interval Regression Loss
        :param gamma_reg: the weight of Regularization (default: (2-norm)^2)
        """
        super(FuzzyLearningMachine, self).__init__()
        self.__fea_ext_net = fea_ext_net
        self.__fsr_net = fsr_net

        self.__ce_loss = nn.CrossEntropyLoss()
        self.__int_reg_loss = IntervalRegressionLoss()
        self.__reg_term = RegularizationTerm()
        self.__gamma_ce = gamma_ce
        self.__gamma_int_reg = gamma_int_reg
        self.__gamma_reg = gamma_reg

        self.__optimizer = None

    def ToStr(self):
        rep_str = self.__repr__()
        rep_str += '\n'
        rep_str += 'gamma_ce=\t' + str(self.__gamma_ce) + '\n'
        rep_str += 'gamma_int_reg=\t' + str(self.__gamma_int_reg) + '\n'
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

    def __Update_parameters(self, batch_data, preprocessing: ImagePreprocessing = None,
                            device=torch.device('cpu'), times=10):
        LB = batch_data['LB_Mat'].to(device)  # shape=[batch_size, batch_size]
        UB = batch_data['UB_Mat'].to(device)  # shape=[batch_size, batch_size]
        y = batch_data['y'].to(device)  # shape=[batch_size]
        X = batch_data['X'].to(device)  # shape=[batch_size, sample.shape]
        if preprocessing is not None:
            X = preprocessing(X)

        ce_loss = -np.inf
        int_reg_loss = -np.inf
        reg = -np.inf
        for _ in range(times):
            self.__optimizer.zero_grad()
            H = self.__fea_ext_net.ComputeLatentRepresentation(X=X)  # shape=[batch_size, latent_fea_num]
            S = self.__fsr_net.ComputeFSR_Mat(H=H)  # shape=[batch_size, batch_size]

            int_reg_loss = self.__int_reg_loss.ComputeLoss(FSR_Mat=S, LB_Mat=LB, UB_Mat=UB)
            total = self.__gamma_int_reg * int_reg_loss
            int_reg_loss = int_reg_loss.clone().detach().cpu().numpy()

            if self.__gamma_ce > 0.0:
                ce_loss = self.__ce_loss(input=H, target=y)
                total += self.__gamma_ce * ce_loss
                ce_loss = ce_loss.clone().detach().cpu().numpy()

            if self.__gamma_reg > 0.0:
                reg = self.__reg_term(m=self)
                total += self.__gamma_reg * reg
                reg = reg.clone().detach().cpu().numpy()

            total.backward()
            self.__optimizer.step()

        return {'ce_loss': ce_loss, 'int_reg_loss': int_reg_loss, 'reg': reg}

    def Train_with_log(self, train_data_set: MyDataSet, preprocessing: ImagePreprocessing = None,
                       max_epoch_num=100, batch_size=64, lr=1e-3, how_often_decay_lr=10000000, decay_factor=1.0,
                       test_preprocessing: ImagePreprocessing = None, test_data_set: MyDataSet = None,
                       class_exemplar_num=5, res_dir=None, device=torch.device('cpu')):
        self.train()
        self.__optimizer = optim.Adam(params=self.parameters(), lr=lr)
        log = None
        if res_dir is not None:
            log = open(file=res_dir + 'Train-log.txt', mode='w')
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
        for epo_ind in range(1, max_epoch_num + 1):
            ce_loss_list = []
            int_reg_loss_list = []
            reg_list = []
            train_data_set.InitEpoch(batch_size=batch_size)
            while True:
                batch_data = train_data_set.NextBatch()
                res = train_data_set.GetClassRelationInterval(class_label_arr=batch_data['y'])
                batch_data['LB_Mat'] = res['LB_Mat']
                batch_data['UB_Mat'] = res['UB_Mat']
                res = self.__Update_parameters(batch_data=batch_data, preprocessing=preprocessing, device=device)
                ce_loss_list.append(res['ce_loss'])
                int_reg_loss_list.append(res['int_reg_loss'])
                reg_list.append(res['reg'])
                if batch_data['is_last_batch']:
                    break
            ce_loss = np.mean(np.array(ce_loss_list))
            int_reg_loss = np.mean(np.array(int_reg_loss_list))
            reg = np.mean(np.array(reg_list))

            acc = -1
            acc2 = -1
            if test_data_set is not None and epo_ind % 10 == 0:
                res = self.SelectExemplarPredict(train_data_set=train_data_set, test_dataset=test_data_set,
                                                 preprocessing=test_preprocessing, device=device, class_exemplar_num=class_exemplar_num)
                # res2 = self.Predict(test_dataset=test_data_set, preprocessing=test_preprocessing, device=device)
                acc = res['acc']
                if acc > best_acc:
                    best_acc = acc
                    if res_dir is not None:
                        self.Save2File(file_name=res_dir + 'best-model.model')
                        Tool.SavePredictResult(res=res, file_name=res_dir + 'best-predict-result.txt')

            # res1 = self.SelectExemplarPredict(train_data_set=train_data_set, test_dataset=copy(train_data_set),
            #                                   preprocessing=test_preprocessing, device=device, class_exemplar_num=class_exemplar_num)
            res1 = {'acc': -1}
            print(Tool.TimeStr() + '\tepoch=\t' + str(epo_ind) + '\tend,' +
                  # '\tce_loss=\t' + str(ce_loss) +
                  '\tint_reg_loss=\t' + str(int_reg_loss) +
                  # '\treg=\t' + str(reg) +
                  '\texemplar-test-acc=\t' + str(acc) +
                  '\texemplar-train-acc=\t' + str(res1['acc'])
                  # '\targmax-test-acc=\t' + str(acc2)
                  )
            if log is not None:
                log.write(Tool.TimeStr() + '\tepoch=\t' + str(epo_ind) + '\tend,' +
                          '\tce_loss=\t' + str(ce_loss) +
                          '\tint_reg_loss=\t' + str(int_reg_loss) +
                          '\treg=\t' + str(reg) +
                          '\texemplar-test-acc=\t' + str(acc) +
                          '\targmax-test-acc=\t' + str(acc2) + '\n')
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

    def Predict(self, test_dataset: MyDataSet, preprocessing: ImagePreprocessing = None,
                batch_size=10000, device=torch.device('cpu')):
        is_train_state = self.training
        self.eval()
        test_dataset.InitEpoch(batch_size=batch_size, do_shuffle=False)
        pre_y = []
        while True:
            batch_data = test_dataset.NextBatch()
            X = batch_data['X'].to(device)
            if preprocessing is not None:
                X = preprocessing(X)
            H = self.__fea_ext_net.ComputeLatentRepresentation(X=X)
            y = torch.argmax(input=H, dim=1)
            pre_y = pre_y + list(y.detach().clone().cpu().numpy())
            if batch_data['is_last_batch']:
                break
        pre_y = np.array(pre_y)
        true_y = test_dataset.GerAll_y()
        acc = np.mean((pre_y == true_y) + 0.0)
        if is_train_state:
            self.train()
        return {'acc': acc, 'pre_y': pre_y, 'true_y': true_y}

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

    def SelectExemplar(self, data_set: MyDataSet, class_exemplar_num=5, batch_size=10000,
                       preprocessing: ImagePreprocessing = None, device=torch.device('cpu')):
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
                sam_score = np.sum(FSR_Mat_c, axis=1)  # shape=[c_sam_num]
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
                              class_exemplar_num=5, batch_size=10000,
                              preprocessing: ImagePreprocessing = None, device=torch.device('cpu')):
        exe_data = self.SelectExemplar(data_set=train_data_set, class_exemplar_num=class_exemplar_num,
                                       batch_size=batch_size, preprocessing=preprocessing, device=device)
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
