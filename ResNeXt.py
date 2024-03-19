"""
ResNet和ResNeXt , 这一网络用于设计更深的网络; ResNeXt在ResNet基础上借鉴了GoogLeNet, 分出多个相同头最后Concat拼接.

He等人认为ResNet中每个附加层都应该更容易地将恒等函数包含为其元素之一；
由于这个问题诞生的解决方案就是Residual Block残差块的提出，RNN，Transformer，GNN都借鉴了这个思路。

注意Residual Block和torch.cat()相似, 但是完全不同。Residual Block是使用的+, 这意味着两个相加的部分必须shape相同;
而cat()是在channel深度方向上进行拼接, 需要除了channel之外的shape相同.

其中有两个有意思的函数如下：
- `nn.AdaptiveAvgPool2d((1, 1))`: 用池化层，将输出特征矩阵输出到指定大小; 他和AvgPool2d的最大区别在于AvgPool2d需要指定kernel_size, stride等参数, 最后的输出特征矩阵由这些参数决定。

是 PyTorch 中的一个层，用于在二维输入数据上执行自适应平均池化操作。它的作用是将输入的二维数据进行池化，使得输出的大小固定为指定的大小。
具体来说，nn.AdaptiveAvgPool2d((1, 1)) 的作用是将输入的二维数据进行池化，使得输出的大小为 (1, 1)。这意味着无论输入的尺寸是多少，最终输出的尺寸都将是 (1, 1)。 

- `self.net.add_module(name: str, module: Module) -> None`: 用于在自定义模型时动态的地向模型中添加子模块。

而遍历每一层的名字和shape则可以用        
for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

也如同apply中用法:
for module in self.children():
            module.apply(fn)
"""
import torch
from torch import nn
from torch.nn import functional as F
import core


"""
搭建一个residual block
"""
class ResNeXtBlock(nn.Module):  #@save
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)


blk = ResNeXtBlock(32, 16, 1)
X = torch.randn(4, 32, 96, 96)
print(blk(X).shape)


"""
搭建一个ResNet
"""
class ResNet(core.Classifier):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

@core.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(ResNeXtBlock(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(ResNeXtBlock(num_channels))
    return nn.Sequential(*blk)

@core.add_to_class(ResNet)
def __init__(self, arch, lr=0.1, num_classes=10):
    super(ResNet, self).__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(self.b1())
    for i, b in enumerate(arch):
        self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
    self.net.add_module('last', nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        nn.LazyLinear(num_classes)))
    self.net.apply(core.init_cnn)

class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),
                       lr, num_classes)

ResNet18().layer_summary((1, 1, 96, 96))


"""

training
"""
import dataset_utils
model = ResNet18(lr=0.01, num_classes=21)
trainer = core.Trainer_GPU(max_epochs=10, num_gpus=1)
data = dataset_utils.DataM(batch_size=1, resize=(100, 100))
model.apply_init([next(iter(data.get_dataloader(True)))[0]], core.init_cnn)
trainer.fit(model, data)