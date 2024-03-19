## x.1 Usage


文件组织形式如下：

```
Data
|----FTIR
      |---- 1_data_mapping.csv
      |---- 1
            |---- G-actin_1.dat
            |---- G-actin_2.dat
            |---- ...
      |---- 2
            |---- F-actin_1.dat
            |---- F-actin_2.dat
            |---- ...
      |---- ...  
```

原始数据存放在FTIR下面，具有多个类别，文件名1, 2, ...就是类别标签target；该类别下具有多种文件，如G-actin_1.dat就是signal；signal和target是一一对应的关系；

但是增加了一种新的自动化划分脚本，当你的文件命名为`xxx_target1_.dat`会自动根据`target1`将类别分为1等...

使用`core1.1.py`便可以成功运行

1. 先运行`preprocess.py`再运行`ResNet.py`就行


## x.2 工作日志

0319

已经完成，但是有更多的部分等待后面完成，包括多种指标，多种loss，将train和validation部分分离等。



3. 代码书写

- BCE Loss 用的是Long, 参考`https://blog.csdn.net/BetrayFree/article/details/133927378`



---

20240318

1. 文献调研

2. 环境搭建

3. 代码书写

- 使用一维信号绘图
- 使用`d2l`范式
 
- 将数据更改后，发现有时候读取的信号长度到不了[1, 1000]，将缺少的地方用0进行填充。
- 成功预测不同温度下的蛋白结构。




4. 展望

- 现在loss很大，期望增加预处理阶段，只提取需要的波数的数据，将数据堆叠成.npy数据格式；再根据npy数据格式求出一些统计值如均值，方差，对整个数据集归一化；
- 使用natsort在预处理时候进行排序；
- 了解zarr存储数据；
- 了解特征融合；
- 希望用多线程处理数据预处理；
- 期望用GPU进行网络训练；
- 期望增加一些评价指标；如MSE；SSIM，PSNR等；
- 期望使用.yml修改参数；
- 期望将代码重构为包含Trainer的格式；
- 期望代码能够使用importlib自动导入网络模型；
- `tqdm`, `seaborn`

