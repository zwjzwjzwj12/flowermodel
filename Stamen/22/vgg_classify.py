# vgg_classify.py

import torch
import argparse

from torch.utils.data import DataLoader

import vgg_dataset
import vgg_model

from PIL import Image
from torchvision.transforms import ToTensor

class_names = ["0", "1", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26",  "27"]
label_array = ["非供液不足", "供液不足", "气体影响", "游动阀漏失", "固定阀漏失", "双阀漏失",
               "上碰泵", "下碰泵", "抽油杆断脱", "柱塞脱出工作筒", "稠油", "油井出砂", "错误功图"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str,
                        default="E:\\aboutjiqixuexi\\gybz_16\\test_dataset_genai_area_zuizhong\\0\\650_26_16_15_17_113.7_油井出砂_DXY926X9.png",
                        help='images path')
    parser.add_argument('-model_path', type=str,
                        default="./checkpoint_area_200\\epoch_29_train_dataset_genai_area_zuizhong-best-acc_0.9994012291483758.pth",
                        help='model path')
    parser.add_argument('-img_size', type=int, default=224, help='the size of image, mutiple of 32')
    parser.add_argument('-num_class', type=int, default=1, help='the number of class')
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
    img = img.unsqueeze(0)  # 增加批次维度
    if opt.gpu:
        img = img.cuda()
    output = net(img)

    output_gailv, _ = torch.max(output, 1)
    # torch.max() 函数，用于在指定维度上找到张量中的最大值及其对应的索引。
    print(output_gailv)
    if output_gailv >= 0.5:
        label = 1
    else:
        label = 0
    gailv = float(output_gailv)
    print("预测结果为：{}".format(label_array[label]))
    print("供液不足的概率：{}".format(float(output_gailv)))
    print("非供液不足的概率：{}".format(float(1-output_gailv)))
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
