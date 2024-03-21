import pandas as pd, torch, torchvision
import os 
import core
import cv2
import numpy as np
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

class DataM(core.DataModule):
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28), download=False, root="/home/yingmuzhi/_learning/d2l/data",
                 data_csv_path="/home/yingmuzhi/SpecML2/data/crop2/1_data_mapping.csv",
                 train_data_csv_path="/home/yingmuzhi/SpecML2/data/train/1_data_mapping.csv",
                 val_data_csv_path="/home/yingmuzhi/SpecML2/data/val/1_data_mapping.csv",
                 num_workers=0,):
        super().__init__()
        self.save_hyperparameters()
        # trans = transforms.Compose([transforms.Resize(resize),
        #                             transforms.ToTensor()])
        trans = None
        # train_set
        data_csv_path = train_data_csv_path
        self.train = DiyDataset(
            root=self.root, train=True, transform=trans, download=download, data_csv_path=data_csv_path)
        # validation_set
        data_csv_path = val_data_csv_path
        self.val = DiyDataset(
            root=self.root, train=False, transform=trans, download=download, data_csv_path=data_csv_path)

    def text_labels(self, indices):
        """Return text labels.
    
        Defined in :numref:`sec_fashion_mnist`"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        """Defined in :numref:`sec_fashion_mnist`"""
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                           num_workers=self.num_workers)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        """Defined in :numref:`sec_fashion_mnist`"""
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1), nrows, ncols, titles=labels)

def generate_dataset_csv(folder_path: str, save_path: str=None):
    """
    intro:
        generate data csv file.
    args:
        :param str folder_path: where data store.
        :param str save_path: save csv file path.
    return:
        :param str save_path:
    """
    data_mapping = {}
    if not save_path:
        save_path = os.path.join(folder_path, "1_data_mapping.csv")

    for target in os.listdir(folder_path):
        target_path = os.path.join(folder_path, target)

        # find wether is directory or file
        if os.path.isdir(target_path):
            # traversal files in directory
            for signal_name in os.listdir(target_path):
                signal_path = os.path.join(target_path, signal_name)

                # mapping
                data_mapping[signal_path] = target

    df = pd.DataFrame(list(data_mapping.items()), columns=["signal_path", "target_value"])

    # save as .csv file
    df.to_csv(save_path, index=False)
    print("Data and label mapping saved to 'data_mapping.csv'")
    return save_path 

def read_one_signal(file_path: str, 
                    trans, 
                    CUTTING_LENGTH: int=1000):
    """
    intro:
        read one 1D signal
    args:
        :param str file_path: the file path to read data.
        :param int CUTTING_LENGTH: the length of data u want read from file.
    return:
        :param torch.Tensor: shape is [channel, 1D data]
    """
    # img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # normalize per pic
    mean = np.mean(img)
    std = np.std(img)
    img = ( img - mean ) / std

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    # permuted_tensor = img_tensor.permute(2, 0, 1)
    permuted_tensor = img_tensor

    # img_tensor = trans(img) # dtype == float32
    return permuted_tensor

def read_one_target(target_value: int,
                    trans,):
    """
    intro:
        return target value.
    args:
        :param int target_value:
    return:
        :param torch.Tensor target:
    """
    # img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(target_value, cv2.IMREAD_GRAYSCALE)

    # normalize per pic
    mean = np.mean(img)
    std = np.std(img)
    img = ( img - mean ) / std

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    # permuted_tensor = img_tensor.permute(2, 0, 1)
    permuted_tensor = img_tensor
    
    # img_tensor = trans(img) # dtype == float32
    return permuted_tensor

class DiyDataset(torch.utils.data.Dataset, core.HyperParameters):
    def __init__(self, 
                 train: bool,
                 transform: object,
                 data_csv_path: str,
                 download: bool=False, 
                 root: str="./",
                 ):
        super().__init__()
        self.save_hyperparameters()
        df_data = pd.read_csv(data_csv_path)
        self.signals = df_data.loc[:, "signal_path"].tolist()
        self.targets = df_data.loc[:, "signal_path"].tolist()
        # self.targets = df_data.loc[:, "target_value"].tolist()
        self.trans = transform
        pass

    def __getitem__(self, index):
        signal = read_one_signal(self.signals[index], self.trans)
        target = read_one_target(self.targets[index], self.trans)
        return signal, target
        pass

    def __len__(self):
        return len(self.signals)
        pass


if __name__=="__main__":
    folder_path = "/home/yingmuzhi/SpecML2/data/crop2"
    save_path = None
    data_csv_path = generate_dataset_csv(folder_path=folder_path, save_path=save_path)

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            ])
    dataset = DiyDataset(train=True, transform=trans, data_csv_path=data_csv_path)
    a, b = dataset[0]
    print(a.shape, a.dtype, b.shape, b.dtype)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    print(next(iter(dataloader)))

    pass