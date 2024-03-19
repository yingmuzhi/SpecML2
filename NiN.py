"""
NiN网络成功引入了1*1的结构，形成了CNN中类似全连接层的作用，减少了参数量；
        同时由于全使用CNN，避免了全连接层，这意味着**输入图片大小可以任意大小**
"""
from torch import nn
import core

def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())

class NiN(core.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            nin_block(num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
        self.net.apply(core.init_cnn)

NiN().layer_summary((1, 1, 224, 224))


"""training"""
import dataset_utils
model = NiN(lr=0.05, num_classes=21)
trainer = core.Trainer_GPU(max_epochs=10, num_gpus=1)
data = dataset_utils.DataM(batch_size=1, resize=(100, 100))
model.apply_init([next(iter(data.get_dataloader(True)))[0]], core.init_cnn)
trainer.fit(model, data)