'''
3DUnet
5 layers

there has Big, Median and tiny net, depending on the numbers of parameters.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def lambda_attention(input_tensor, channels):
    batch = input_tensor.shape[0]
    height = width = input_tensor.shape[-1]
    input_tensor_channel = input_tensor.shape[1]
    channels -= input_tensor_channel
    
    empty_tensor = torch.zeros(batch, channels, height, width).to("cuda")
    # empty_tensor = torch.zeros(batch, channels, height, width)
    output_tensor = torch.cat([input_tensor, empty_tensor], dim=1)

    return output_tensor


# DOWNSAMPLE
class ConvBlock(torch.nn.Module):
    def padding_zero(shpae):
        pass

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_f=0.2, first_in=False):
        super(ConvBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.activ0 = nn.ReLU()

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activ1 = nn.ReLU()

        if not first_in:
            self.pool = nn.AvgPool2d(kernel_size=(2, 2))   # 2, 2, 0
        else:
            self.pool = None
        self.residual = nn.Conv2d(in_channels, out_channels, 1, 1, 0) # residual always 1, 1, 0

        self.dropout = nn.Dropout(p=dropout_f)


    def forward(self, x, attention_channel):
        if self.pool is not None:
            temp = self.pool(x)
            residual = self.residual(temp)  # conv
            # residual = lambda_attention(temp, attention_channel)    # lambda_attention
        else:
            residual = self.residual(x)     # conv
            # residual = lambda_attention(x, attention_channel)       # lambda_attention

        x = self.conv0(x)
        x = self.activ0(x)
        x = self.bn0(x)
        x = self.dropout(x)
        
        x = self.conv1(x)
        x = self.activ1(x)
        x = self.bn1(x)
        x = self.dropout(x)

        x += residual
    
        return x


# UPSAMPLE
class DeconvBlock(torch.nn.Module):
    def bilinear_upsample(self, x, scale_factor=2):
        upsampled = F.interpolate(x, scale_factor=scale_factor, mode='bilinear')
        return upsampled

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, padding, dropout_f=0.2):
        super(DeconvBlock, self).__init__()
        # self.conv0 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride, padding)   # conv
        self.conv0 = self.bilinear_upsample     # bilinear

        self.conv1 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.activ1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.activ2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(p=dropout_f)

        self.residule = nn.Conv2d(mid_channels, out_channels, 1, 1, 0)  # 1, 1, 0

    def forward(self, x, y):
        x = self.conv0(x)
        x = torch.cat([x, y], 1)
        
        temp = self.residule(x)

        x = self.conv1(x)
        x = self.activ1(x)
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.activ2(x)
        x = self.bn2(x)
        x = self.dropout(x)

        x += temp

        return x


# big
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # # Multimodal
        # self.multi = ConvBlock(4, 1, kernel_size=1, stride=1, padding=0)

        # in
        self.in_conv = ConvBlock(1, 16, 3, 1, 1, first_in=True)

        # # Encoder
        # self.conv1 = ConvBlock(16, 32, 2, 2, 0)
        # self.conv2 = ConvBlock(32, 64, 2, 2, 0)
        # self.conv3 = ConvBlock(64, 128, 2, 2, 0)
        # self.conv4 = ConvBlock(128, 256, 2, 2, 0)

        # # Decoder
        # self.deconv0 = DeconvBlock(256, 384, 128, 2, 2, 0)
        # self.deconv1 = DeconvBlock(128, 192, 64, 2, 2, 0)
        # self.deconv2 = DeconvBlock(64, 96, 32, 2, 2, 0)
        # self.deconv3 = DeconvBlock(32, 48, 16, 2, 2, 0)
        
        # Encoder
        self.conv1 = ConvBlock(16, 32, 3, 2, 1)
        self.conv2 = ConvBlock(32, 64, 3, 2, 1)
        self.conv3 = ConvBlock(64, 128, 3, 2, 1)
        self.conv4 = ConvBlock(128, 256, 3, 2, 1)

        # Decoder
        self.deconv0 = DeconvBlock(256, 384, 128, 3, 2, 1)
        self.deconv1 = DeconvBlock(128, 192, 64, 3, 2, 1)
        self.deconv2 = DeconvBlock(64, 96, 32, 3, 2, 1)
        self.deconv3 = DeconvBlock(32, 48, 16, 3, 2, 1)
        
        self.out_conv = nn.Conv2d(16, 1, 1, 1, 0)
        # self.out_lambda = nn.Conv2d(1, 1, 1, 1, 0)
        # self.out_activ = nn.Identity()

        self.ce_layer = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.LazyConv2d(100, (1, 1)), torch.nn.Flatten(), torch.nn.LazyLinear(21))



    
    def forward(self, x):
        # # multimodal
        # x = self.multi(x)

        # in
        enc0 = self.in_conv(x, 16)

        # Encoder
        enc1 = self.conv1(enc0, 32)
        enc2 = self.conv2(enc1, 64)
        enc3 = self.conv3(enc2, 128)
        enc4 = self.conv4(enc3, 256)

        # self.embed = torch.Tensor(enc4.size()).copy_(enc4)
        self.embed = enc4
        self.embed2 = self.ce_layer(self.embed)
        

        # Decoder with skip-connections
        dec0 = self.deconv0(enc4, enc3)
        dec1 = self.deconv1(dec0, enc2)
        dec2 = self.deconv2(dec1, enc1)
        dec3 = self.deconv3(dec2, enc0)
        
        # out
        out = self.out_conv(dec3)
        # out = self.out_lambda(out)
        # out = self.out_activ(out)

        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)

    def _init_weight(self, m):
        """
        intro:
            weights init.
            There exists two type of init method.
            >>> if isinstance(m, nn.Linear):
            >>>     nn.init.trunc_normal_(m.weight, std=.01)
            >>>     if m.bias is not None:
            >>>         nn.init.zeros_(m.bias)

            >>> elif classname.startswith('Conv'):
            >>>     m.weight.data.normal_(0.0, 0.02)
        
        args:
            :param torch.parameters m: weights.
        """
        classname = m.__class__.__name__


        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # elif isinstance(m, nn.Linear):
        #     nn.init.trunc_normal_(m.weight, std=.01)
        #     if m.bias is not None:
        #         nn.init.zeros_(m.bias)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.zeros_(m.bias)
        #     nn.init.ones_(m.weight)
        # elif classname.startswith('Conv'):
        #     m.weight.data.normal_(0.0, 0.02)
        # elif classname.find('BatchNorm') != -1:
        #     m.weight.data.normal_(1.0, 0.02)
        #     m.bias.data.fill_(0) 


# calculate parameters
# 参考: `https://blog.csdn.net/qq_41979513/article/details/102369396`
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# Net = Generator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
#from torchsummary import summary
class TrainSet(Dataset):
    def __init__(self, X, Y):
        # 定义好 image 的路径
        self.X, self.Y = X, Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
def main():
    X_tensor = torch.ones((4, 1, 256, 256))
    # X_tensor = torch.ones((4, 1, 100, 100))
    Y_tensor = torch.zeros((4, 1, 256, 256))
    mydataset = TrainSet(X_tensor, Y_tensor)
    train_loader = DataLoader(mydataset, batch_size=2, shuffle=True)

    net=Net()
    print(net)
    net.apply(net._init_weight)
    import torch.nn as nn
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

    print(get_parameter_number(net))
    # 3) Training loop
    for epoch in range(10):
        for i, (X, y) in enumerate(train_loader):
            # predict = forward pass with our model
            pred = net(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch={},i={}'.format(epoch,i))

import core, dataset_utils_256

class Net(core.Classifier):
    def __init__(self, lr, plot_train_per_epoch=2, plot_valid_per_epoch=1,):
        super().__init__(plot_train_per_epoch, plot_valid_per_epoch)
        self.save_hyperparameters()
        self.net = Generator()
    
    def loss(self, Y_hat, Y, averaged=True):
        fn = nn.MSELoss()
        
        loss2 = F.cross_entropy(self.net.embed2, Y, reduction='mean')
        return fn(Y_hat, Y), loss2
    
    def BCE_loss(self, Y_hat, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        Y_hat = torch.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = torch.reshape(Y, (-1,))
        Y = Y.type(torch.long)
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        Y = batch[-1]
        
        loss_value = self.BCE_loss(Y_hat, Y)
        self.metrics_loss.append(loss_value)

    
class Trainer_G(core.Trainer_GPU):
    def prepare_batch(self, batch):
        return super().prepare_batch(batch)

def main():
    model = Net(lr=0.1)
    trainer = core.Trainer_GPU(max_epochs=10, num_gpus=1)
    data = dataset_utils_256.DataM(batch_size=1, resize=(256, 256), train_data_csv_path="/home/yingmuzhi/SpecML2/data/train_256/1_data_mapping.csv", val_data_csv_path="/home/yingmuzhi/SpecML2/data/val_256/1_data_mapping.csv")
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], core.init_cnn)
    trainer.fit(model, data)
    pass
if __name__ == '__main__':
    main()
    # import os, sys
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # from torchsummary import summary
    # model = Net()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device=device)
    # input_tensor: tuple = (1, 256, 256)
    # summary(model, input_size=input_tensor)   