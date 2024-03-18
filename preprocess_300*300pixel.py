import matplotlib.pyplot as plt
import numpy as np
import cv2

def resize_pic(pic_path: str,
               resize_path: str,):
    # 读取图像文件
    img = cv2.imread(pic_path)

    # 调整图像大小为 224x224 像素
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)  # 使用INTER_AREA插值方法

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
    inch_size = (2, 2)
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
    plt.xticks(ticks=range(1000, 4001, 500),    # 设置刻度间隔为200
               fontsize=4, 
            #    rotation=45
               )
    plt.xlim(1000, 4000)  # 设置x轴范围
    plt.yticks(ticks=(np.arange(-1, 1.25, 0.25)),
               fontsize=4,
               )
    plt.ylim(-1., 1.)
    plt.grid(True,
             linewidth=0.5
             )  # 添加网格线
    # plt.show()
    # wanna save 224 * 224 pixel png
    # pixel_size = 224
    # dpi = int(pixel_size / inch_size)
    plt.savefig(pic_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    data_path = "/home/yingmuzhi/SpecML2/data/origin_data/Insulin+0.2MCu.0.dat"
    pic_path = "/home/yingmuzhi/SpecML2/data/png/1.png"
    resize_path = "/home/yingmuzhi/SpecML2/data/png/2.png"

    
    convert_one_data2pic(data_path, pic_path)   # 300 * 300 is okay
    #resize_pic(pic_path, resize_path)           # 224 * 224 is not okay