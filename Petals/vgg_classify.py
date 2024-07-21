# vgg_classify.py
import os

import numpy as np
import torch
import argparse

from torch.utils.data import DataLoader

import vgg_dataset
import vgg_model

from PIL import Image
from torchvision.transforms import ToTensor


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


class_names = ["15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"]
label_array = ["工况正常", "供液不足", "气体影响", "游动阀漏失", "固定阀漏失", "双阀漏失",
               "上碰泵", "下碰泵", "抽油杆断脱", "柱塞脱出工作筒", "稠油", "油井出砂", "错误功图", "管漏失"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str,
                        default="E:\\aboutjiqixuexi\\pushi_area\\train_dataset_genai_area_zuizhong\\28\\34_28_26_40_40_40.50_管漏失_99.72537994384766_+_26_油井出砂_0.2608802318572998_+_22_下碰泵_0.006906022317707539_CTZ12.png",
                        help='images path')
    parser.add_argument('-model_path', type=str,
                        default="E:\\aboutjiqixuexi\\pushi_area\\checkpoint_area_200\\epoch_54_train_dataset_genai_area_zuizhong-best-acc_0.9692381620407104-jracc_0.9798125624656677-erjracc_0.9992790194664743.pth",
                        help='model path')
    parser.add_argument('-img_size', type=int, default=224, help='the size of image, mutiple of 32')
    parser.add_argument('-num_class', type=int, default=14, help='the number of class')
    parser.add_argument('-gpu', default=True, help='use gpu or not')

    opt = parser.parse_args()

    # initialize vgg
    if opt.gpu:
        net = vgg_model.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net = vgg_model.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class)

    # load model data
    net.load_state_dict(torch.load(opt.model_path))
    net.eval()

    img = Image.open(opt.img_path)
    # if len(img.split()) == 1:
    #     img = img.convert("RGB")
    img = img.resize((opt.img_size, opt.img_size))
    image_to_tensor = ToTensor()
    img = image_to_tensor(img)
    img = img.unsqueeze(0)
    if opt.gpu:
        img = img.cuda()
    output = net(img)
    x = int(float(os.path.basename(opt.img_path).split('_')[5]))  # 获得面积参数
    if x > 600:
        x = 0
    area_shu = float(int(pengzhangfenbu(x, 30))) / 80
    area_l = []
    for i in range(0, 200):
        area_l.append(area_shu)
    # print(area_l)
    area = torch.tensor(area_l)
    if opt.gpu:
        area = area.cuda()

    # 在第二个维度上使用 torch.cat 进行拼接操作
    connect = torch.cat((output, area.unsqueeze(0)), dim=1)
    output = connect
    for fc in net.fc_list:  # 3 FC
        output = fc(output)
    _, indices = torch.max(output, 1)
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100  # 各类别索引列表
    perc = percentage[int(indices)].item()  # 最大概率
    result = class_names[indices]  # 最大概率数字标签

    # print('predicted:', result)
    # 获取前三个最大的数据
    top_three_values = sorted(percentage, reverse=True)[:3]
    # 获取前三个最大数据的索引
    top_three_indices = sorted(range(len(percentage)), key=lambda i: percentage[i], reverse=True)[:3]

    # perc = percentage[int(indices)].item()  # 将最大的概率拿出赋值给perc
    result_1 = class_names[top_three_indices[0]]  # 将最大概率的数字标签拿到
    perc_1 = percentage[top_three_indices[0]]  # 将最大概率的拿到

    result_2 = class_names[top_three_indices[1]]  # 将第二概率的数字标签拿到
    perc_2 = percentage[top_three_indices[1]]  # 将第二的概率拿到
    result_3 = class_names[top_three_indices[2]]  # 将第三概率的数字标签拿到
    perc_3 = percentage[top_three_indices[2]]  # 将第三的概率拿到

    wenzilabel_1 = label_array[int(result_1) - 15]  # 概率最大的文字标签
    wenzilabel_2 = label_array[int(result_2) - 15]
    wenzilabel_3 = label_array[int(result_3) - 15]
    print("预测结果1：代号：{}，文字标签：{}，概率：{}%".format(result_1, wenzilabel_1, perc_1))
    print("预测结果2：代号：{}，文字标签：{}，概率：{}%".format(result_2, wenzilabel_2, perc_2))
    print("预测结果3：代号：{}，文字标签：{}，概率：{}%".format(result_3, wenzilabel_3, perc_3))
    # 这段代码涉及了使用预训练模型进行图像分类的过程。让我逐步解释每一部分的含义：
    # img = img.unsqueeze(0)
    # 这行代码在图像张量的维度上增加一个维度，将其从形状(C, H, W)变为(1, C, H, W)。
    # 这是因为模型通常接受批次输入，所以需要在第  一维度上添加一个批次维度。
    # _, indices = torch.max(output, 1)
    # 这行代码使用torch.max()函数获取输出张量output在第1维度（类别维度）上的最大值和对应的索引。indices保存了每个样本的预测类别索引。
    # percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
    # perc = percentage[int(indices)].item()
    # 这部分代码使用torch.nn.functional.softmax()函数对输出张量output进行 softmax 操作，将输出转换为概率分布。然后，从概率分布中获取预测类别的概率百分比。
    # result = class_names[indices]
    # 这行代码根据预测的类别索引indices，从class_names列表中获取对应的类别标签。
    # print('predicted:', result)
    # 这行代码打印输出预测的类别标签。
