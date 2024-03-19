"""
请注意，虽然LeNet在28*28*60000的训练集和1w张测试集表现良好，但其使用的网络浅层，使用激活函数Sigmoid（如果
模型参数没有正确初始化，sigmoid函数可能在正区间内获得几乎为0的梯度），使用了权重衰减（L2正则坏），在更大的数据集上并没有良好表现。

引入2012在ImageNet上表现良好的AlexNet。它的优势在于更加深的网络结构，引入了ReLU激活函数，使用GPU训练加快速度，在原有
权重衰减基础上增加了dropout层，使用翻转·裁剪和颜色变化等图像增强功能。
"""
from torch import nn
import core


# model
class AlexNet(core.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes))
        self.net.apply(core.init_cnn)

AlexNet().layer_summary((1, 1, 224, 224))


# train
import dataset_utils
model = AlexNet(lr=0.01, num_classes=21)
data = dataset_utils.DataM(batch_size=1, resize=(100, 100))
trainer = core.Trainer_GPU(max_epochs=10, num_gpus=1)
trainer.fit(model, data)