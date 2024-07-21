# vgg_classify.py
import glob
import os
import shutil

import torch
import argparse

from torch.utils.data import DataLoader

import vgg_dataset
import vgg_model

from PIL import Image
from torchvision.transforms import ToTensor

class_names = ["15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26",  "27"]
label_array = ["工况正常", "供液不足", "气体影响", "游动阀漏失", "固定阀漏失", "双阀漏失",
               "上碰泵", "下碰泵", "抽油杆断脱", "柱塞脱出工作筒", "稠油", "油井出砂", "错误功图"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str, default="./train_dataset_genai_area_zuizhong\\16", help='images path')
    parser.add_argument('-gongkuang', type=str, default="上碰泵", help='本分类器的工况类别')
    parser.add_argument('-new_img_path', type=str, default="./make_label_test_sig(shang_chu33_xia_chu50)_16_e45", help='new images path')
    parser.add_argument('-model_path', type=str, default="E:\\aboutjiqixuexi\\spb_21\\checkpoint_wei_sig(shang_chu33_xia_chu50)\\epoch_45_train_dataset_genai_area_zuizhong-best-acc_1.0.pth", help='model path')
    parser.add_argument('-img_size', type=int, default=224, help='the size of image, mutiple of 32')
    parser.add_argument('-num_class', type=int, default=1, help='the number of class')
    parser.add_argument('-gpu', default=True, help='use gpu or not')

    opt = parser.parse_args()

    # # 从文件夹获取图片数据
    # data_list = list()
    # for file in os.listdir("dataset_test/"):
    #     data_list.append(os.path.join("dataset_test/", file))
    # print("Load {} Data Successfully!".format("dataset_test"))

    # initialize vgg
    if opt.gpu:
        net = vgg_model.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net = vgg_model.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class)

    # load model data
    net.load_state_dict(torch.load(opt.model_path))
    net.eval()
    # net.eval()是用于将神经网络模型切换到评估模式的函数调用。

    # 使用 glob 模块匹配文件夹中的所有 png 文件
    csv_files = glob.glob(os.path.join(opt.img_path, '*.png'))  # 这个列表是跟文件夹倒序的
    num_batches = len(csv_files)  # 总共的组数
    for file in os.listdir(opt.img_path):  # 从文件夹获取图片数据的文件名
        file_path = os.path.join(opt.img_path, file)  # 组合文件路径
        img = Image.open(file_path)  # 获取文件路径下的图片数据
        img = img.resize((opt.img_size, opt.img_size))  # 缩放为长为self.img_size，宽为self.img_size的图
        image_to_tensor = ToTensor()
        img = image_to_tensor(img)

        # if len(img.split()) == 1:
        #     img = img.convert("RGB")
        img = img.unsqueeze(0)
        # 这种操作通常用于在处理单张图像时，将其转换为批次大小为1的张量，以满足模型输入的要求。
        if opt.gpu:
            img = img.cuda()
        output = net(img)

        output_gailv, _ = torch.max(output, 1)

        # 这段代码涉及了使用预训练模型进行图像分类的过程。让我逐步解释每一部分的含义：
        # img = img.unsqueeze(0)
        # 这行代码在图像张量的维度上增加一个维度，将其从形状(C, H, W)变为(1, C, H, W)。
        # 这是因为模型通常接受批次输入，所以需要在第一维度上添加一个批次维度。
        # _, indices = torch.max(output, 1)
        # 这行代码使用torch.max()函数获取输出张量output在第1维度（类别维度）上的最大值和对应的索引。indices保存了每个样本的预测类别索引。
        # percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        # perc = percentage[int(indices)].item()
        # 这部分代码使用torch.nn.functional.softmax()函数对输出张量output进行 softmax 操作，将输出转换为概率分布。然后，从概率分布中获取预测类别的概率百分比。
        # result = class_names[indices]
        # 这行代码根据预测的类别索引indices，从class_names列表中获取对应的类别标签。
        # print('predicted:', result)
        # 这行代码打印输出预测的类别标签。
        print(output_gailv)

        print(file)

        # 拆分文件名
        parts = file.split("_")

        if output_gailv >= 0.5:
            label = 1
        else:
            label = 0
        # 修改第二个位置的部分
        parts.insert(6, "m{}+{}概率：{:.4f}".format(label, opt.gongkuang, float(output_gailv)))

        # 组合新的文件名
        new_filename = "_".join(parts)
        print(new_filename)
        # 原文件路径
        original_path = file_path

        # 新文件路径
        new_path_fu = opt.new_img_path
        if not os.path.exists(new_path_fu):
            os.makedirs(new_path_fu)
        new_path = os.path.join(new_path_fu, new_filename)
        #  复制文件并重命名保存到新路径
        shutil.copy2(original_path, new_path)
        # # 重命名文件
        # os.rename(original_path, new_path)



