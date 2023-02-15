import torch.nn as nn
from functions import ReverseLayerF


class CNNModel(nn.Module):
    """
    类内部实现generator和class_classifier
    generator本质是一个图片用自编码器，用于进行源域图片的领域变换
    class_classifier用于完成变换后的源域样本分类
    """
    def __init__(self):
        super(CNNModel, self).__init__()

        self.generator = nn.Sequential()
        self.generator.add_module('g_fc1', nn.Linear(28 * 28 * 3, 500))
        #self.generator.add_module('g_bn1', nn.BatchNorm1d(500))
        self.generator.add_module('g_relu1', nn.ReLU(True))
        self.generator.add_module('g_fc2', nn.Linear(500, 300))
        #self.generator.add_module('g_bn2', nn.BatchNorm1d(300))
        self.generator.add_module('g_relu2', nn.ReLU(True))
        self.generator.add_module('g_fc3', nn.Linear(300, 100))
        self.generator.add_module('g_relu3', nn.ReLU(True))
        self.generator.add_module('g_fc4', nn.Linear(100, 300))
        self.generator.add_module('g_relu4', nn.ReLU(True))
        self.generator.add_module('g_fc5', nn.Linear(300, 500))
        self.generator.add_module('g_relu5', nn.ReLU(True))
        self.generator.add_module('g_fc6', nn.Linear(500, 28 * 28 * 3))
        #self.generator.add_module('g_bn6', nn.BatchNorm1d(28 * 28))
        self.generator.add_module('g_relu6', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.class_classifier.add_module('f_bn1', nn.BatchNorm2d(64))
        self.class_classifier.add_module('f_pool1', nn.MaxPool2d(2))
        self.class_classifier.add_module('f_relu1', nn.ReLU(True))
        self.class_classifier.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.class_classifier.add_module('f_bn2', nn.BatchNorm2d(50))
        self.class_classifier.add_module('f_drop1', nn.Dropout2d())
        self.class_classifier.add_module('f_pool2', nn.MaxPool2d(2))
        self.class_classifier.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data):
        input_data = input_data.view(-1, 28 * 28)
        feature = self.generator(input_data)

        class_output = self.class_classifier(feature)

        return feature, class_output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        input_data = ReverseLayerF.apply(input_data, alpha)
        domain_output = self.domain_classifier(input_data)

        return domain_output
