import numpy as np
import torch
from torch import nn
import hdf5storage

from Core.DataSet import TableDataSet, ClassSpaceInformation
from Core.FeatureExtractionNetwork import FCN
from Core.FuzzyQuotientSpace import FuzzyRelation
from Core.FuzzySimilarityRelationNetwork import SigmoidCos
from Core.FuzzyLearningMachine import FuzzyLearningMachine
from Core.ClassSimilarity import ClassSimilarity
from Core.Kit import Tool
from Core.Tree import ClassTree
from Core.EvaluationMeasures import HieClaEvaMea

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')

data_dir = '/root/DataSet/ZSL-DataSet/'
res_dir = '/root/Result@ZSLDataSet/'
data_name = 'AWA1'

# 1. Construct Class Tree and Class FER
alpha = 0.2
beta = 0.8
# class_inf = ClassSpaceInformation(cls_space_inf_txt_file=data_dir + data_name + '/class_space_information.txt')
# class_tree = ClassTree.ConstructClassTree(class_inf=class_inf)
# print('depth= ' + str(class_tree.GetDepth()))
# print('node number= ' + str(class_tree.GetNodeNum()))
# print('leaf number= ' + str(class_tree.GetLeafNum()))
# hcm = HieClaEvaMea(class_tree=class_tree)

# 2. Load Data Set
data_set = TableDataSet(Xy_mat_file=data_dir + data_name + '/' + data_name + '_Xy.mat')
""" A-->Graph """
A = hdf5storage.loadmat(file_name=data_dir + data_name + '/' + data_name + '_A.mat')['A']
S = ClassSimilarity.ComputeSimilarity(A=A, sim_type='Cos')
LUB = ClassSimilarity.ClassGraph2LU_Bound(S=S, alpha=alpha, beta=beta, min_sim=1e-3)
data_set.SetClassFSR_LUB2(LB=LUB['LB'], UB=LUB['UB'])

is_train = hdf5storage.loadmat(file_name=data_dir + data_name + '/' + data_name + '_IsTrain5fold.mat')['is_train'] == 1
# 3. Training and Test
for f in [3]:  # [0, 1, 2, 3, 4]:
    two_data = data_set.SplitTrainTest(is_train=is_train[f, :])
    tr_data_set = two_data['train_data_set']
    te_data_set = two_data['test_data_set']
    print('f= ' + str(f) + ' train(' + str(tr_data_set.GetSampleNum()) + ')<-->test(' + str(te_data_set.GetSampleNum()) + ')')

    beg_str = Tool.TimeStr2()
    # res_prefix = res_dir + 'FLM_Cos(A)_' + data_name + '@fold' + str(f) + '@' + beg_str + '_'
    Tool.SetRandSeed(rand_seed=1234)

    fcn = FCN(in_features=1, out_features=1)
    net = torch.nn.Sequential()
    net.add_module(name='Layer1', module=nn.Linear(in_features=2048, out_features=1024, bias=True))
    net.add_module(name='Layer1-Act', module=nn.Sigmoid())
    net.add_module(name='Layer2', module=nn.Linear(in_features=1024, out_features=512, bias=True))
    # net.add_module(name='Layer2-Act', module=nn.Sigmoid())
    # net.add_module(name='Layer3', module=nn.Linear(in_features=512, out_features=tr_data_set.GetClassNum(), bias=True))
    fcn.SetNet(net=net)
    fcn.apply(FCN.Init)

    m = FuzzyLearningMachine(fea_ext_net=fcn, fsr_net=SigmoidCos(), FPL_alpah=alpha, FPL_beta=beta,
                             gamma_FPL=0.0, gamma_int_reg=1.0, gamma_sim_rank=0.0, gamma_reg=0.0)
    m.to(device=device)
    # m.SaveSetting2File(file_name=res_prefix + 'Setting.txt')
    m.Train_with_log(train_data_set=tr_data_set, max_epoch_num=1000, batch_size=1000, lr=1e-3, how_often_decay_lr=50,
                     decay_factor=0.5, test_data_set=te_data_set, train_log_prefix=None, device=device)
    pre_res = m.SelectExemplarPredict(train_data_set=tr_data_set, test_dataset=te_data_set, device=device)

    # Tool.SavePredictResult(res=pre_res, file_name=res_prefix + 'PredictResult.txt')
    print(Tool.PredictResult2Str(res=pre_res))
