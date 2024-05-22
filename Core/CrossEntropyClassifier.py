import numpy as np
import torch
from torch import nn, optim

from Core.FeatureExtractionNetwork import FeatureExtractionNetwork
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


class CrossEntropyClassifier(nn.Module):
    def __init__(self, fea_ext_net: FeatureExtractionNetwork, gamma_reg=0.0):
        super(CrossEntropyClassifier, self).__init__()
        self.__fea_ext_net = fea_ext_net
        self.__ce_loss = nn.CrossEntropyLoss()
        self.__reg_term = RegularizationTerm()
        self.__gamma_reg = gamma_reg
        self.__optimizer = None

    def __Update_parameters(self, batch_data, preprocessing: ImagePreprocessing = None, device=torch.device('cpu'),
                            times=1):
        X = batch_data['X'].to(device)  # shape=[batch_size, sample.shape]
        y = batch_data['y'].to(device)  # shape=[batch_size]
        if preprocessing is not None:
            X = preprocessing(X)
        loss = -np.inf
        reg = -np.inf
        for _ in range(times):
            self.__optimizer.zero_grad()
            H = self.__fea_ext_net.ComputeLatentRepresentation(X=X)  # shape=[batch_size, latent_fea_num]
            loss = self.__ce_loss(input=H, target=y)
            total = loss
            loss = loss.clone().detach().cpu().numpy()
            if self.__gamma_reg > 0.0:
                reg = self.__reg_term(m=self)
                total += self.__gamma_reg * reg
                reg = reg.clone().detach().cpu().numpy()
            total.backward()
            self.__optimizer.step()

        return {'loss': loss, 'reg': reg}

    def ToStr(self):
        rep_str = self.__repr__()
        rep_str += '\n'
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

    def Train_with_log(self, train_data_set: MyDataSet,
                       preprocessing: ImagePreprocessing = None,
                       test_preprocessing: ImagePreprocessing = None,
                       max_epoch_num=100, batch_size=64, lr=1e-3,
                       how_often_decay_lr=10000000, decay_factor=1.0, test_data_set: MyDataSet = None,
                       res_dir=None, device=torch.device('cpu')):
        self.train()
        self.__optimizer = optim.Adam(params=self.parameters(), lr=lr)
        log = None
        if res_dir is not None:
            log = open(file=res_dir + 'TrainLog.txt', mode='w')
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
            log.flush()

        best_acc = -1
        for epo_ind in range(1, max_epoch_num + 1):
            loss_list = []
            reg_list = []
            train_data_set.InitEpoch(batch_size=batch_size)
            while True:
                batch_data = train_data_set.NextBatch()
                res = self.__Update_parameters(batch_data=batch_data, preprocessing=preprocessing, device=device)
                loss_list.append(res['loss'])
                reg_list.append(res['reg'])
                if batch_data['is_last_batch']:
                    break
            loss = np.mean(np.array(loss_list))
            reg = np.mean(np.array(reg_list))

            acc = -1
            if test_data_set is not None and epo_ind % 1 == 0:
                res = self.Predict(test_dataset=test_data_set, device=device, preprocessing=test_preprocessing)

                acc = res['acc']
                if acc > best_acc:
                    best_acc = acc
                    self.Save2File(file_name=res_dir + 'TrainLogBest.model')
                    Tool.SavePredictResult(res=res, file_name=res_dir + 'TrainLogBestPredictResult.txt')

            print(Tool.TimeStr() + '\tepoch=\t' + str(epo_ind) + '\tend,' +
                  '\tloss=\t' + str(loss) +
                  '\treg=\t' + str(reg) +
                  '\ttest-acc=\t' + str(acc))
            if log is not None:
                log.write(Tool.TimeStr() + '\tepoch=\t' + str(epo_ind) + '\tend,' +
                          '\tloss=\t' + str(loss) +
                          '\treg=\t' + str(reg) +
                          '\ttest-acc=\t' + str(acc) + '\n')
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

    def Predict(self, test_dataset: MyDataSet, preprocessing: ImagePreprocessing = None, batch_size=10000,
                device=torch.device('cpu')):
        self.eval()
        pre_y = []
        test_dataset.InitEpoch(batch_size=batch_size, do_shuffle=False)
        while True:
            batch_data = test_dataset.NextBatch()
            X = batch_data['X'].to(device)
            if preprocessing is not None:
                X = preprocessing(X)
            H = self.__fea_ext_net.ComputeLatentRepresentation(X=X)
            b_pre_y = torch.argmax(input=H, dim=1)
            b_pre_y = list(b_pre_y.detach().clone().cpu().numpy())
            pre_y = pre_y + b_pre_y
            if batch_data['is_last_batch']:
                break

        pre_y = np.array(pre_y)
        te_y = test_dataset.GerAll_y()
        acc = np.mean((pre_y == te_y) + 0.0)
        self.train()
        return {'acc': acc, 'pre_y': pre_y, 'true_y': te_y}

    def Save2File(self, file_name):
        torch.save(obj=self, f=file_name)

    @staticmethod
    def LoadFromFile(file_name, map_location=None):
        model = torch.load(f=file_name, map_location=map_location)
        return model
