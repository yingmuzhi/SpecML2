"""
调参主要是在`convert_one_data2pic`
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import re
import shutil


def generate_one_100pixel_pic(data_path: str,
                              pic_path: str,
                              resize_path: str,
                              crop_path: str):
    convert_one_data2pic(data_path, pic_path)   # 300 * 300 is okay
    # resize_pic(pic_path, resize_path)           # 224 * 224 is not okay, chage resize path
    crop_pic(resize_path, crop_path)


def crop_pic(pic_path: str,
             crop_path: str,):
    # 读取图像文件
    img = cv2.imread(pic_path)

    # 获取图像的尺寸
    height, width, _ = img.shape

    # 计算裁剪的区域, 总大小100*100pixel
    top = max(0, height // 2 - 50)
    bottom = min(height, height // 2 + 50)
    left = max(0, width // 2 - 50)
    right = min(width, width // 2 + 50)

    # 裁剪图像
    cropped_img = img[top:bottom, left:right]

    # 保存裁剪后的图像
    cv2.imwrite(crop_path, cropped_img)


def resize_pic(pic_path: str,
               resize_path: str,):
    # 读取图像文件
    img = cv2.imread(pic_path)

    # 调整图像大小为 224x224 像素
    img_resized = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)  # 使用INTER_AREA插值方法

    # 保存调整大小后的图像
    cv2.imwrite(resize_path, img_resized)

def convert_one_data2pic(data_path: str,
                         pic_path: str):
    """
    intro:
        inch_size和dpi最终会影响你整个图像的大小和成像质量
    """
    # 从文件中加载数据
    data_file = data_path  # 请替换为您的数据文件路径
    x_values = []
    y_values = []
    with open(data_file, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split(','))
            x_values.append(x)
            y_values.append(y)

    # 创建一个新的图像对象并设置大小, 因为后续需要进行图像处理, 224*224pixel大小适配大多数网络，所以修改成正方形大小
    # inch_size = (8, 6)  # 如果设置为8, 6, 一切都会合理起来
    inch_size = (0.8, 0.8)
    plt.figure(figsize=(inch_size[0], inch_size[1]))  # 设置图像大小为 8x8 英寸

    # 绘制二维图像
    # 散点图
    # plt.scatter(x_values, y_values, s=1)  # s为点的大小，可以根据需要调整
    # 折线图
    plt.plot(x_values, y_values, 
            #  marker='o', 
            linestyle='-', 
            linewidth=1, 
            )
    # plt.xlabel('X', fontsize=4)
    # plt.ylabel('Y', fontsize=4)
    # plt.title('png')
    plt.xticks(ticks=range(1500, 1801, 100),    # 设置刻度间隔为200
               fontsize=4, 
            #    rotation=45
               )
    plt.xlim(1500, 1800)  # 设置x轴范围
    plt.yticks(ticks=(np.arange(-0.2, 0.2, 0.2)),
               fontsize=4,
               )
    plt.ylim(-0.05, 0.15)
    plt.tick_params(axis='both', which='major', pad=4)  # 设置刻度标签与坐标轴之间的距离为8个点
    plt.grid(True,
             linewidth=0.5
             )  # 添加网格线

    # 去掉刻度和外边框
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 去除刻度
    plt.axis('off')  # 去除外边框

    # plt.show()
    # wanna save 224 * 224 pixel png
    # pixel_size = 224
    # dpi = int(pixel_size / inch_size)
    plt.savefig(pic_path, dpi=150)
    plt.close()

def find_dat_files(directory):
    dat_files = []
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.dat'):
                dat_files.append(os.path.join(root, file))
                file_names.append(file[:-4])
    return dat_files, file_names

def generate_multi_100pixel_pic(files_path: str, save_path: str):
    origin_data = os.path.join(files_path, "data")
    dat_files, file_names = find_dat_files(files_path)
    for index, data_path in enumerate(dat_files):
        # 检查目录是否存在
        if not os.path.exists(save_path):
            # 如果目录不存在，则创建目录
            os.makedirs(save_path)
            print("目录已创建")

        pic_path = os.path.join(files_path, 'temp', file_names[index]+".png")
        resize_path = pic_path
        crop_path = os.path.join(files_path, 'crop', file_names[index]+'.png')
        generate_one_100pixel_pic(data_path, pic_path, pic_path, crop_path)

def generate_target(directory, save_path):
    # 文件绝对路径
    dat_files = []
    # 用于存储目标字符串的数组
    targets = []
    
    # 正则表达式模式，用于匹配目标字符串
    pattern = r'target(\d+)'
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                dat_files.append(os.path.join(root, file))
                # 正则表达式判断
                match = re.search(pattern, file)
                if match:
                    # 提取到目标字符串，并存储到数组中
                    target = match.group(1)
                    # target; 创建目录名字
                    targets.append((target, os.path.join(save_path, target), os.path.join(directory, file), os.path.join(save_path, target, file)))

    for target, target_dir, source_file, target_file in targets:
        # 检查目录是否存在
        if not os.path.exists(target_dir):
            # 如果目录不存在，则创建目录
            os.makedirs(target_dir)
            print("目录已创建")

            # 复制文件
            shutil.copyfile(source_file, target_file)
        else:
            print("目录已存在")
    return         


# 第一步：注释掉第二步，运行第一步; 用于生成crop2文件夹
if __name__ == "__main__":
    directory = "/home/yingmuzhi/SpecML2/data/crop"
    files_path = "/home/yingmuzhi/SpecML2/data"
    generate_multi_100pixel_pic(files_path, directory)

    directory = "/home/yingmuzhi/SpecML2/data/crop"
    save_path = "/home/yingmuzhi/SpecML2/data/crop2"
    generate_target(directory, save_path)


# 第二步：注释掉第一步，运行第二步; 将生成的crop2文件夹分别改名成train和val文件夹; 下面这步是在train和val文件夹中创建.csv文件; 这一步骤中是根据文件名来生成target标签
# if __name__=="__main__":
#     from dataset_utils import *

#     folder_path = "/home/yingmuzhi/SpecML2/data/train"
#     save_path = None
#     data_csv_path = generate_dataset_csv(folder_path=folder_path, save_path=save_path)

#     folder_path = "/home/yingmuzhi/SpecML2/data/val"
#     save_path = None
#     data_csv_path = generate_dataset_csv(folder_path=folder_path, save_path=save_path)
    

    