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
class_inf = ClassSpaceInformation(cls_space_inf_txt_file=data_dir + data_name + '/' + 'class_space_information.txt')
class_tree = ClassTree.ConstructClassTree(class_inf=class_inf)
print('depth= ' + str(class_tree.GetDepth()))
print('node number= ' + str(class_tree.GetNodeNum()))
print('leaf number= ' + str(class_tree.GetLeafNum()))
hcm = HieClaEvaMea(class_tree=class_tree)

# 2. Load Data Set
data_set = TableDataSet(Xy_mat_file=data_dir + data_name + '/' + data_name + '_Xy.mat')
""" A-->Graph """
A = hdf5storage.loadmat(file_name=data_dir + data_name + '/' + data_name + '_A.mat')['A']
# S = ClassSimilarity.ComputeSimilarity(A=A, sim_type='Cos')
S = ClassSimilarity.WordNetSimilarity(class_inf=class_inf, sim_type='path_similarity')#{'path_similarity',  'lch_similarity', 'wup_similarity'}
S = S/np.max(S)
# S = ClassSimilarity.KNN_Graph(S=S, k=31, knn_type='or')
LUB = ClassSimilarity.ClassGraph2LU_Bound(S=S, alpha=alpha, beta=beta, min_sim=1e-3)
data_set.SetClassFSR_LUB2(LB=LUB['LB'], UB=LUB['UB'])
""" Random Graph """
# S = ClassSimilarity.GenerateRandomSimilarity(max_class_id=data_set.GetMaxClassID())
# LUB = ClassSimilarity.ClassGraph2LU_Bound(S=S, alpha=alpha, beta=beta)
# data_set.SetClassFSR_LUB2(LB=LUB['LB'], UB=LUB['UB'])

is_train = hdf5storage.loadmat(file_name=data_dir + data_name + '/' + data_name + '_IsTrain5fold.mat')['is_train'] == 1
# 3. Training and Test
for f in [0]:  # [0, 1, 2, 3, 4]:
    two_data = data_set.SplitTrainTest(is_train=is_train[f, :])
    tr_data_set = two_data['train_data_set']
    te_data_set = two_data['test_data_set']
    print('f= ' + str(f) + ' train(' + str(tr_data_set.GetSampleNum()) + ')<-->test(' + str(te_data_set.GetSampleNum()) + ')')

    beg_str = Tool.TimeStr2()
    res_prefix = res_dir + 'FLM_CT_' + data_name + '@fold' + str(f) + '@' + beg_str + '_'
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
    m.Train_with_log(train_data_set=tr_data_set, max_epoch_num=200, batch_size=1000, lr=1e-3, how_often_decay_lr=200,
                     decay_factor=0.9, test_data_set=te_data_set, train_log_prefix=None, device=device)
    pre_res = m.SelectExemplarPredict(train_data_set=tr_data_set, test_dataset=te_data_set, device=device)
    # TIE = hcm.TreeInducedError(true_y_arr=pre_res['true_y'], pre_y_arr=pre_res['pre_y'])
    # PRF = hcm.Hie_Precision_Recall_F1(true_y_arr=pre_res['true_y'], pre_y_arr=pre_res['pre_y'])
    # pre_res['TIE'] = TIE
    # pre_res['hie_precision'] = PRF['hie_precision']
    # pre_res['hie_recall'] = PRF['hie_recall']
    # pre_res['hie_F1'] = PRF['hie_F1']

    # Tool.SavePredictResult(res=pre_res, file_name=res_prefix + 'PredictResult.txt')
    print(Tool.PredictResult2Str(res=pre_res))

# 用A构图
# acc=	0.8939000648929266
# TIE=	0.7972096041531473
# hie_precision=	0.9549567064451879
# hie_recall=	0.9561439903020045
# hie_F1=	0.9547001341620497

# 用A构图
# max_epoch_num=500, batch_size=64
# FPL_alpah=0.2, FPL_beta=0.8,
# gamma_FPL=1.0, gamma_sim_rank=1.0, gamma_reg=0.0
# acc=	0.8786502271252433
# TIE=	0.9432186891628812
# hie_precision=	0.9468467734407059
# hie_recall=	0.9450834307015358
# hie_F1=	0.9448204005962998

# 随机构图
# 2048->1024->512
# max_epoch_num=500, batch_size=500
# f= 0 train(12257)<-->test(3082)
# acc=	0.8932511356262167
# TIE=	0.8225178455548345
# hie_precision=	0.9548784148110885
# hie_recall=	0.9521880703296991
# hie_F1=	0.9525581434947096
#
# f= 1 train(12266)<-->test(3073)
# acc=	0.885128538887081
# TIE=	0.8760169215750081
# hie_precision=	0.9490256001726877
# hie_recall=	0.9468561625126871
# hie_F1=	0.9469500623776054
#
# f= 2 train(12273)<-->test(3066)
# acc=	0.878016960208741
# TIE=	0.9422700587084148
# hie_precision=	0.9471679390588393
# hie_recall=	0.9442627958038916
# hie_F1=	0.9447565490377251
#
# f= 3 train(12276)<-->test(3063)
# acc=	0.8952007835455436
# TIE=	0.8459027097616716
# hie_precision=	0.9533525639304287
# hie_recall=	0.9504541047880912
# hie_F1=	0.9510022662906586
#
# f= 4 train(12284)<-->test(3055)
# acc=	0.8851063829787233
# TIE=	0.9037643207855974
# hie_precision=	0.9481252966891266
# hie_recall=	0.9449985416474779
# hie_F1=	0.9456790307084397

# 不构图，模糊容忍损失 0.2,  0.8
# 2048->1024->512
# max_epoch_num=500, batch_size=500