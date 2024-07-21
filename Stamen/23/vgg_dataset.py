# vgg_dataset.py

import os

import numpy as np
import torch
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torchvision.models as models

# 定义变换后的 sigmoid 函数
def transformed_sigmoid(x):
    return 1 / (1 + np.exp(-x * 15)) - 0.5


# 定义膨胀分布函数
def pengzhangfenbu(x, center):
    qujian1 = [-0.5, 0]
    q1len = qujian1[1] - qujian1[0]
    daqujian1 = [0, center]
    dq1len = daqujian1[1] - daqujian1[0]
    qujian2 = [0, 0.5]
    q2len = qujian2[1] - qujian2[0]
    daqujian2 = [center, 255]
    dq2len = daqujian2[1] - daqujian2[0]

    if x < center:
        y = transformed_sigmoid((x - center) / (dq1len / q1len)) * (dq1len / q1len) + center
        return y
    else:
        y = transformed_sigmoid((x - center) / (dq2len / q2len)) * (dq2len / q2len) + center
        return y


class MyDataset(Dataset):
    def __init__(self, type, img_size, data_dir):
        self.name2label = {"0": 0, "1": 1, "17": 2, "18": 3, "19": 4, "20": 5, "21": 6, "22": 7, "23": 8, "24": 9,
                           "25": 10, "26": 11, "27": 12, "28": 13, "30": 14, "40": 15}
        self.img_size = img_size
        self.data_dir = data_dir
        self.data_list = list()

        for class_dir in range(0, 2):  # 已经分好了文件夹的数据集导入方式
            for file in os.listdir(os.path.join(self.data_dir, "{}".format(class_dir))):  # listdir 函数来遍历指定目录中的文件和文件夹。
                #  列表中的元素都是文件名
                # listdir 函数返回指定目录中的所有文件和文件夹的列表。
                self.data_list.append(os.path.join(self.data_dir, "{}".format(class_dir), file))  # 路径要认真写，仔细检查

                # data_list.append向列表里面添加图片
                # os.path.join路径拼接
                # 这句代码将文件路径中的图片，挨个放入data_list里面
        # # 未分文件夹的数据集的导入方式
        # for file in os.listdir(self.data_dir):  # listdir 函数来遍历指定目录中的文件和文件夹。
        #     # listdir 函数返回指定目录中的所有文件和文件夹的列表。
        #     self.data_list.append(os.path.join(self.data_dir, file))  # 路径要认真写，仔细检查
        #     # data_list.append向列表里面添加图片
        #     # os.path.join路径拼接
        #     # 这句代码将文件路径中的图片，挨个放入data_list里面
        print("Load {} Data Successfully!".format(type))
        # 表示放入完毕

    def __len__(self):
        return len(self.data_list)
        # 表示data_list的长度，图片数量

    def __getitem__(self, item):  # 数据的序号
        file = self.data_list[item]  #
        img = Image.open(file)
        # if len(img.split()) == 1:
        #     # img.split()：该方法用于将图像拆分为各个通道。对于彩色图像，它将返回R、G、B三个通道的图像。对于灰度图像，它将返回单个通道的图像。
        #     # len(img.split())：这行代码计算了图像拆分后得到的通道数。如果通道数为1，表示图像是灰度图。
        #     # img.convert("RGB")：如果图像是灰度图（通道数为1），则使用convert()方法将其转换为RGB图像。这是因为后续的处理可能需要使用彩色图像的特性。
        #     img = img.convert("RGB")
        img = img.resize((self.img_size, self.img_size))  # 缩放为长为self.img_size，宽为self.img_size的图
        # 只获取单标签：
        # label = self.name2label[os.path.basename(file).split('_')[1]]

        # 获取标签
        label = float(file.split("\\")[-2])  # 以文件夹的名字来当做标签
        # os.path.basename(file)：这是使用os.path模块中的basename函数，传入一个文件路径 file，返回文件的基本名称（不包括路径部分）。
        # 例如，如果file为"/path/to/example_file.txt"，那么os.path.basename(file)将返回"example_file.txt"。
        # .split('_')：这是对基本文件名进行字符串分割操作，使用下划线作为分隔符。通过调用split()方法并传入下划线作为分隔符，将基本文件名分割为多个部分，并返回一个列表。
        # [1]：这是对分割后的列表进行索引操作，提取第一个元素。索引从0开始，因此[1]表示列表中的第二个元素。
        image_to_tensor = ToTensor()
        img = image_to_tensor(img)
        x = int(float(os.path.basename(file).split('_')[5]))  # 获得面积参数
        if x > 600:
            x = 0
        area_shu = float(int(pengzhangfenbu(x, 30))) / 80
        area_l = []
        for i in range(0, 200):
            area_l.append(area_shu)
        # print(area_l)
        area = torch.tensor(area_l)
        label = tensor(label)  # 将标签也转化为张量
        return img, label, area
