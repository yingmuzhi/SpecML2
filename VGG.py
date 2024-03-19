"""
vgg通过引入多个3*3的卷积，达到了更深网络的同时，减少了参数量。

VGG中3*3的块已经形成了范式: CNN: kernel_size = (3, 3), padding = 1, stride = (1, 1)
                        +POOLING: kernel_size = (2, 2), stride = (2, 2)

VGG这种将多层组装成一个块的思想，以及3*3结构给后续带来了很多启发
"""
from torch import nn
# from d2l import torch as d2l
import core


"""
VGG Blocks:
定义一个block块, 包含了N个CNN块+ReLU 后接1个MaxPool2d.
"""
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)


"""VGG: Blocks + Classifier"""
class VGG(core.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes))
        self.net.apply(core.init_cnn)

temp = VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)))
temp.layer_summary((1, 1, 224, 224))


"""training"""
import dataset_utils
model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01, num_classes=21)  # id(model) != id(temp)
trainer = core.Trainer_GPU(max_epochs=10, num_gpus=1)
data = dataset_utils.DataM(batch_size=1, resize=(100, 100))
model.apply_init([next(iter(data.get_dataloader(True)))[0]], core.init_cnn) # model.net.apply(d2l.init_cnn)
trainer.fit(model, data)