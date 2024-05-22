import torch
from torch import nn
import hdf5storage

from Core.DataSet import TableDataSet, ClassSpaceInformation
from Core.FeatureExtractionNetwork import FCN
from Core.CrossEntropyClassifier import CrossEntropyClassifier
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

# 1. Construct Class Tree
# class_inf = ClassSpaceInformation(cls_space_inf_txt_file='/root/cjbDisk/DataSet/ZSL-DataSet/CUB/class_space_information.txt')
# class_tree = ClassTree.ConstructClassTree(class_inf=class_inf)
# hcm = HieClaEvaMea(class_tree=class_tree)
# print('depth= ' + str(class_tree.GetDepth()))
# print('node number= ' + str(class_tree.GetNodeNum()))
# print('leaf number= ' + str(class_tree.GetLeafNum()))


# 2. Load Data Set
data_set = TableDataSet(Xy_mat_file=data_dir + data_name + '/' + data_name + '_Xy.mat')
is_train = hdf5storage.loadmat(file_name=data_dir + data_name + '/' + data_name + '_IsTrain5fold.mat')['is_train'] == 1

# 3. Training and Test
for f in [0]:  # [0, 1, 2, 3, 4]
    two_data = data_set.SplitTrainTest(is_train=is_train[f, :])
    tr_data_set = two_data['train_data_set']
    te_data_set = two_data['test_data_set']
    print('f= ' + str(f) + ' train(' + str(tr_data_set.GetSampleNum()) + ')<-->test(' + str(te_data_set.GetSampleNum()) + ')')

    beg_str = Tool.TimeStr2()
    # res_prefix = res_dir + 'CE_CT_' + data_name + '@fold' + str(f) + '@' + beg_str + '_'
    Tool.SetRandSeed(rand_seed=1234)

    fcn = FCN(in_features=1, out_features=1)
    net = torch.nn.Sequential()
    net.add_module(name='Layer1',
                   module=nn.Linear(in_features=2048, out_features=tr_data_set.GetSampleNum(), bias=True))
    # net.add_module(name='Layer1-Act', module=nn.Sigmoid())
    # net.add_module(name='Layer2', module=nn.Linear(in_features=4096, out_features=1024, bias=True))
    # net.add_module(name='Layer2-Act', module=nn.Sigmoid())
    # net.add_module(name='Layer3', module=nn.Linear(in_features=512, out_features=tr_data_set.GetClassNum(), bias=True))
    fcn.SetNet(net=net)
    fcn.apply(FCN.Init)

    m = CrossEntropyClassifier(fea_ext_net=fcn, gamma_reg=0.0)
    m.to(device=device)
    # m.SaveSetting2File(file_name=res_prefix + 'Setting.txt')
    m.Train_with_log(train_data_set=tr_data_set, max_epoch_num=1000, batch_size=256, lr=1e-3,
                     how_often_decay_lr=10000, decay_factor=0.9, test_data_set=te_data_set,
                     res_dir=None, device=device)
    pre_res = m.Predict(test_dataset=te_data_set, device=device)

    # TIE = hcm.TreeInducedError(true_y_arr=pre_res['true_y'], pre_y_arr=pre_res['pre_y'])
    # PRF = hcm.Hie_Precision_Recall_F1(true_y_arr=pre_res['true_y'], pre_y_arr=pre_res['pre_y'])
    # pre_res['TIE'] = TIE
    # pre_res['hie_precision'] = PRF['hie_precision']
    # pre_res['hie_recall'] = PRF['hie_recall']
    # pre_res['hie_F1'] = PRF['hie_F1']

    # Tool.SavePredictResult(res=pre_res, file_name=res_prefix + 'PredictResult.txt')
    print(Tool.PredictResult2Str(res=pre_res))


# max_epoch_num=100, batch_size=256
# acc=	0.890330953926022
# TIE=	0.8400389357560026
# hie_precision=	0.9524902394994867
# hie_recall=	0.9506430209058372
# hie_F1=	0.9506709815377016

# max_epoch_num=500, batch_size=256
# acc=	0.8867618429591174
# TIE=	0.8744321868916288
# hie_precision=	0.9512934043028138
# hie_recall=	0.9490532144328382
# hie_F1=	0.9492879441483052

# max_epoch_num=1000, batch_size=256
# acc=	0.8857884490590525
# TIE=	0.8929266709928618
# hie_precision=	0.9501365158203251
# hie_recall=	0.9481519237846298
# hie_F1=	0.9482562893850356