import random
import time

import numpy as np
import torch


class Tool(object):
    @staticmethod
    def LinearFunction(x1, y1, x2, y2, x):
        k = (y1 - y2) / (x1 - x2)
        y = k * (x - x1) + y1
        return y

    @staticmethod
    def SetRandSeed(rand_seed):
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)
        random.seed(rand_seed)
        np.random.seed(rand_seed)

    @staticmethod
    def TimeStr():
        return time.strftime('%y-%m-%d@%H:%M:%S', time.localtime())

    @staticmethod
    def TimeStr2():
        return time.strftime('%y-%m-%d@%H-%M-%S', time.localtime())

    @staticmethod
    def PredictResult2Str(res):
        s = 'acc=\t' + str(res['acc']) + '\n'
        if res.__contains__('TIE'):
            s += 'TIE=\t' + str(res['TIE']) + '\n'
        if res.__contains__('hie_precision'):
            s += 'hie_precision=\t' + str(res['hie_precision']) + '\n'
        if res.__contains__('hie_recall'):
            s += 'hie_recall=\t' + str(res['hie_recall']) + '\n'
        if res.__contains__('hie_F1'):
            s += 'hie_F1=\t' + str(res['hie_F1']) + '\n'
        return s

    @staticmethod
    def SavePredictResult(res, file_name):
        """

        :param res: {'acc': acc, 'pre_y': pre_y, 'true_y': te_y}
        :param file_name:
        :return:
        """
        f = open(file=file_name, mode='w')
        f.write('acc=\t' + str(res['acc']) + '\n')
        if res.__contains__('TIE'):
            f.write('TIE=\t' + str(res['TIE']) + '\n')
        if res.__contains__('hie_precision'):
            f.write('hie_precision=\t' + str(res['hie_precision']) + '\n')
        if res.__contains__('hie_recall'):
            f.write('hie_recall=\t' + str(res['hie_recall']) + '\n')
        if res.__contains__('hie_F1'):
            f.write('hie_F1=\t' + str(res['hie_F1']) + '\n')

        pre_y = res['pre_y']
        true_y = res['true_y']
        sam_num = pre_y.shape[0]
        f.write('Sample-ID(from 1)\tTrue-y\tPredict_Exemplar-y\n')
        for i in range(sam_num):
            if true_y[i] != pre_y[i]:
                f.write(str(i + 1) + '\t' + str(true_y[i]) + '\t' + str(pre_y[i]) + '\n')
        f.close()

    @staticmethod
    def ReadPredictResult(txt_file_name, num_test_sam):
        pre_res = {}
        fr = open(file=txt_file_name, mode='r')
        line = fr.readline()  # acc=	0.6028552887735237
        line = line.replace('\n', '')
        line = line.replace('acc=\t', '')
        pre_res['acc'] = float(line)

        fr.readline()  # Sample-ID(from 1)	True-y	Predict_Exemplar-y

        true_y = np.zeros(shape=[num_test_sam], dtype=np.int64)
        pre_y = np.zeros(shape=[num_test_sam], dtype=np.int64)
        ind = 0
        line = fr.readline()  # 1	11	16
        while line != '':
            line = line.replace('\n', '')
            int_arr = line.split('\t')
            true_y[ind] = int(int_arr[1])
            pre_y[ind] = int(int_arr[2])
            ind += 1
            line = fr.readline()
        fr.close()

        # Check
        acc = np.mean((pre_y == true_y) + 0.0)
        if acc != pre_res['acc']:
            raise Exception('acc != pre_res[\'acc\']')
        # Check

        pre_res['true_y'] = true_y
        pre_res['pre_y'] = pre_y
        return pre_res

    @staticmethod
    def PrintInformation(m: torch.nn.Module):
        print(str(type(m)))
        print(str(type(m)) + '========Structure======================')
        print(m)
        print(str(type(m)) + '========Parameters======================')
        paras = m.named_parameters()
        i = 0
        para_num = 0
        for name, para in paras:
            print('ID=\t' + str(i) +
                  '\tname=\t' + name +
                  '\ttype=\t' + str(type(para)) +
                  '\tshape=\t' + str(para.shape) +
                  '\tnumel=\t' + str(para.numel()))
            para_num += para.numel()
            i += 1
        print('number of parameters=\t' + str(para_num))


class List(object):
    def __init__(self):
        self.__data = []

    def IsEmpty(self):
        return len(self.__data) == 0

    # ================ Queue ====================
    def InQueue(self, ele):
        self.__data.append(ele)

    def InQueue_list(self, ele_list):
        self.__data += ele_list

    def OutQueue(self):
        ele = self.__data[0]
        self.__data = self.__data[1:len(self.__data)]
        return ele

    def PeekQueue(self):
        return self.__data[0]

    # ================ Stack ====================
    def InStack(self, ele):
        self.__data.append(ele)

    def InStack_list(self, ele_list):
        self.__data += ele_list

    def OutStack(self):
        ele = self.__data[-1]
        len1 = len(self.__data) - 1
        self.__data = self.__data[0: len1]
        return ele

    def PeekStack(self):
        return self.__data[-1]
