import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision as tv

from Core.VisionTransformer import VisionTransformer, PatchEmbed

"""
fea: Feature
num: Number
"""


class FeatureExtractionNetwork(nn.Module):
    def __init__(self):
        super(FeatureExtractionNetwork, self).__init__()

    def ComputeLatentRepresentation(self, X: Tensor) -> Tensor:
        """

        :param X: shape=[batch_size, feature_num] or shape=[batch_size, channel_num, image_width, image_high]
        :return H: shape=[batch_size, latent_feature_num]
        """
        pass


class FCN(FeatureExtractionNetwork):
    def __init__(self, in_features, out_features):
        super(FCN, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module(name='Layer1', module=nn.Linear(in_features=in_features, out_features=1024, bias=True))
        self.net.add_module(name='Layer1-Act', module=nn.ReLU())
        # self.net.add_module(name='Layer1-Act', module=nn.Sigmoid())
        self.net.add_module(name='Layer2', module=nn.Linear(in_features=1024, out_features=512, bias=True))
        self.net.add_module(name='Layer2-Act', module=nn.ReLU())
        # self.net.add_module(name='Layer2-Act', module=nn.Sigmoid())
        self.net.add_module(name='Layer3', module=nn.Linear(in_features=512, out_features=out_features, bias=True))
        self.apply(FCN.Init)

    def SetNet(self, net):
        self.net = net

    def ComputeLatentRepresentation(self, X: Tensor) -> Tensor:
        return self.net(X)

    @staticmethod
    def Init(m):
        # print(type(m))
        if type(m) == nn.Linear:
            # print(type(m))
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)


class ModelM7(FeatureExtractionNetwork):
    """
    References
    [1] Sanghyeon An, Minjun Lee, Sanglee Park, Heerin Yang, Jungmin So.
        An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition.
        https://doi.org/10.48550/arXiv.2008.10400
    The code is download from: https://github.com/ansh941/MnistSimpleCNN
    """

    def __init__(self):
        super(ModelM7, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, 7, bias=False)  # output becomes 22x22
        self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 7, bias=False)  # output becomes 16x16
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 144, 7, bias=False)  # output becomes 10x10
        self.conv3_bn = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144, 192, 7, bias=False)  # output becomes 4x4
        self.conv4_bn = nn.BatchNorm2d(192)
        self.fc1 = nn.Linear(3072, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)

    def ComputeLatentRepresentation(self, X: Tensor) -> Tensor:
        x = (X - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0, 2, 3, 1), 1)
        h = self.fc1_bn(self.fc1(flat1))
        return h

    @staticmethod
    def init_fn(m):
        # print(type(m))
        if type(m) == nn.modules.conv.Conv2d:
            nn.init.kaiming_normal_(tensor=m.weight, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
            # print(type(m))
        elif type(m) == nn.modules.linear.Linear:
            nn.init.kaiming_normal_(tensor=m.weight, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
            # print(type(m))


class LeNet(FeatureExtractionNetwork):
    """
    LeCun, Y., Boser, B.E., Denker, J.S., Henderson, D., Howard, R.E., Hubbard, W.E., and Jackel, L.D.
    Handwritten digit recognition with a back-propagation network.
    In Advances in Neural Information Processing Systems, Denver, Colorado, USA, pp. 396–404, 1989.
    """

    def __init__(self, image_channel, latent_fea_num=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(image_channel, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        if image_channel == 1:  # MNIST
            self.fc1 = nn.Linear(256, 120)
        else:  # CIFAR10
            self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()

        # remove classification head
        # self.fc3 = nn.Linear(84, num_classes)
        # self.relu5 = nn.ReLU()

        self.fc = nn.Linear(84, latent_fea_num)

    def ComputeLatentRepresentation(self, X: Tensor) -> Tensor:
        """

        :param X: shape=[sam_num, image_channel, image_width, image_high]
        :return H: shape=[sam_num, latent_fea_num]
        """
        y = self.conv1(X)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        h = self.relu4(y)
        # __y = self.fc3(h)
        # __y = self.relu5(__y)
        h = self.fc(h)  # shape=[n_sam, n_hidden_feature]
        return h


class ResNet(FeatureExtractionNetwork):
    """
    He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition.
    In IEEE Conference on Computer Vision and Pattern Recognition, Las Vegas, NV, USA, pp. 770–778, 2016.
    """

    def __init__(self, in_channels, latent_fea_num, net_type='resnet18'):
        super(ResNet, self).__init__()
        if net_type == 'resnet18':
            net_list = list(tv.models.resnet18(pretrained=False).children())
            self.fc = nn.Linear(in_features=512, out_features=latent_fea_num, bias=True)
        elif net_type == 'resnet50':
            net_list = list(tv.models.resnet50(pretrained=False).children())
            self.fc = nn.Linear(in_features=2048, out_features=latent_fea_num, bias=True)
        else:
            raise Exception('net_type=' + net_type + ' is not supported.')
        net = net_list[:-1]  # remove classification head
        # print(res_net_list1[0])
        # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if in_channels == 1:
            net[0] = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=(3, 3),
                               bias=False)

        self.res_net = nn.Sequential(*net)
        # self.dropout = nn.Dropout(0.5)

        self.apply(fn=ResNet.Init)

    def ComputeLatentRepresentation(self, X: Tensor) -> Tensor:
        """

        :param X: shape=[n_sam, image_channel, image_width, image_high]
        :return H: # shape=[n_sam, n_hidden_feature]
        """
        h = torch.flatten(self.res_net(X), 1)
        # h = self.dropout(h)
        h = self.fc(h)  # shape=[n_sam, n_hidden_feature]
        return h

    @staticmethod
    def Init(m):
        # print(type(m))
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # print(type(m))
            nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def Init_Kaiming(m):
        # print(type(m))
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # print(type(m))
            nn.init.kaiming_normal_(tensor=m.weight, nonlinearity='relu')


class ResNet_back_bone(FeatureExtractionNetwork):
    """
    He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition.
    In IEEE Conference on Computer Vision and Pattern Recognition, Las Vegas, NV, USA, pp. 770–778, 2016.
    """

    def __init__(self, in_channels, net_type='resnet18', init_fn=ResNet.Init):
        super(ResNet_back_bone, self).__init__()
        if net_type == 'resnet18':
            net_list = list(tv.models.resnet18(pretrained=False).children())
        elif net_type == 'resnet50':
            net_list = list(tv.models.resnet50(pretrained=False).children())
        else:
            raise Exception('net_type=' + net_type + ' is not supported.')
        net = net_list[:-1]  # remove classification head
        # print(res_net_list1[0])
        # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if in_channels == 1:
            net[0] = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=(3, 3),
                               bias=False)
        self.res_net = nn.Sequential(*net)

        if init_fn is not None:
            self.apply(fn=init_fn)

    def ComputeLatentRepresentation(self, X: Tensor) -> Tensor:
        """

        :param X: shape=[n_sam, image_channel, image_width, image_high]
        :return H: # shape=[n_sam, n_hidden_feature]
        """
        h = torch.flatten(self.res_net(X), 1)
        return h


class ViT(FeatureExtractionNetwork):
    """
    Alexey Dosovitskiy, Lucas Beyer and Alexander Kolesnikov, et. al.
    An image is worth 16x16 words:transformers for image recognition at scale.
    International Conference on Learning Representations, 2021.
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, latent_fea_num=10, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super(ViT, self).__init__()
        self.net = VisionTransformer(img_size=img_size, patch_size=patch_size, in_c=in_c, num_classes=latent_fea_num,
                                     embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale, representation_size=representation_size,
                                     distilled=distilled, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                                     drop_path_ratio=drop_path_ratio, embed_layer=embed_layer, norm_layer=norm_layer,
                                     act_layer=act_layer)

    def ComputeLatentRepresentation(self, X: Tensor) -> Tensor:
        return self.net(X)
