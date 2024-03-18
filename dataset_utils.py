import pandas as pd, torch, torchvision
import os 
import core
import cv2


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
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    img_tensor = trans(img) # dtype == float32
    return img_tensor

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
    target = torch.tensor(target_value, dtype=torch.float32).unsqueeze(0)
    return target

class DiyDataset(torch.utils.data.Dataset, core.HyperParameters):
    def __init__(self, 
                 root: str,
                 train: bool,
                 download: bool, 
                 transform: object,
                 data_csv_path: str):
        df_data = pd.read_csv(data_csv_path)
        self.signals = df_data.loc[:, "signal_path"].tolist()
        self.targets = df_data.loc[:, "target_value"].tolist()
        self.trans = transform
        pass

    def __getitem__(self, index):
        self.signal = read_one_signal(self.signals[index], self.trans)
        self.target = read_one_target(self.targets[index], self.trans)
        return self.signal, self.target
        pass

    def __len__(self):
        len(self.targets)
        pass


if __name__=="__main__":
    folder_path = "/home/yingmuzhi/SpecML2/data/crop2"
    save_path = None
    data_csv_path = generate_dataset_csv(folder_path=folder_path, save_path=save_path)

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            ])
    dataset = DiyDataset(train=True, transform=trans, data_csv_path=data_csv_path)
    dataset[0]


    pass