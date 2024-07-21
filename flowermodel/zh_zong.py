import glob
import os
import shutil

import numpy as np
import torch
import argparse

from torch.utils.data import DataLoader

import vgg_dataset
import vgg_model

from PIL import Image
from torchvision.transforms import ToTensor

import vgg_dataset
import vgg_model
import vgg_model_16
import vgg_model_17
import vgg_model_18
import vgg_model_19
import vgg_model_20
import vgg_model_21
import vgg_model_22
import vgg_model_23
import vgg_model_24
import vgg_model_25
import vgg_model_26
import vgg_model_pushi_area

class_names = ["15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"]
label_array = ["工况正常", "供液不足", "气体影响", "游动阀漏失", "固定阀漏失", "双阀漏失",
               "上碰泵", "下碰泵", "抽油杆断脱", "柱塞脱出工作筒", "稠油", "油井出砂", "错误功图", "管漏失"]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str, default="test_dataset_genai_area_zuizhong",
                        help='images path')
    parser.add_argument('-new_img_path', type=str,
                        default="./makelabel_zh_zong_test_addgls",
                        help='new images path')
    parser.add_argument('-model_path_pushi_area', type=str,
                        default="./moxing_path\\pushi_area_epoch_54_train_dataset_genai_area_zuizhong-best-acc_0.9692381620407104-jracc_0.9798125624656677-erjracc_0.9992790194664743.pth",
                        help='model path')
    parser.add_argument('-model_path_16', type=str,
                        default="./moxing_path\\gybz_16_epoch_57_train_dataset_genai_area_zuizhong-best-acc_0.9961945031712474.pth",
                        help='model path')
    parser.add_argument('-model_path_17', type=str,
                        default="./moxing_path\\qtyx_17_epoch_38_train_dataset_genai_area_zuizhong-best-acc_0.9943820224719101.pth",
                        help='model path')
    parser.add_argument('-model_path_18', type=str,
                        default="./moxing_path\\ydfls_18_epoch_51_train_dataset_genai_area_zuizhong-best-acc_0.9992592592592593.pth",
                        help='model path')
    parser.add_argument('-model_path_19', type=str,
                        default="./moxing_path\\gdfls_19_epoch_57_train_dataset_genai_area_zuizhong-best-acc_0.998062015503876.pth",
                        help='model path')
    parser.add_argument('-model_path_20', type=str,
                        default="./moxing_path\\sfls_20_epoch_68_train_dataset_genai_area_zuizhong-best-acc_0.9956521739130435.pth",
                        help='model path')
    parser.add_argument('-model_path_21', type=str,
                        default="./moxing_path\\spb_21_epoch_60_train_dataset_genai_area_zuizhong-best-acc_1.0.pth",
                        help='model path')
    parser.add_argument('-model_path_22', type=str,
                        default="./moxing_path\\xpb_22_epoch_54_train_dataset_genai_area_zuizhong-best-acc_0.9983552631578947.pth",
                        help='model path')
    parser.add_argument('-model_path_23', type=str,
                        default="./moxing_path\\cygdt_23_epoch_5_train_dataset_genai_area_zuizhong-best-acc_0.9909722222222223.pth",
                        help='model path')
    parser.add_argument('-model_path_24', type=str,
                        default="./moxing_path\\zstcgzt_24_epoch_16_train_dataset_genai_area_zuizhong-best-acc_1.0.pth",
                        help='model path')
    parser.add_argument('-model_path_25', type=str,
                        default="./moxing_path\\cy_25_epoch_36_train_dataset_genai_area_zuizhong-best-acc_0.9978880675818373.pth",
                        help='model path')
    parser.add_argument('-model_path_26', type=str,
                        default="./moxing_path\\yjcs_26_epoch_16_train_dataset_genai_area_zuizhong-best-acc_1.0.pth",
                        help='model path')
    parser.add_argument('-img_size', type=int, default=224, help='the size of image, mutiple of 32')
    parser.add_argument('-num_class', type=int, default=1, help='the number of class')
    parser.add_argument('-gpu', default=True, help='use gpu or not')

    opt = parser.parse_args()

    # initialize vgg
    if opt.gpu:
        net_pushi_area = vgg_model_pushi_area.VGG(img_size=opt.img_size, input_channel=1, num_class=14).cuda()
        net_16 = vgg_model_16.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_17 = vgg_model_17.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_18 = vgg_model_18.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_19 = vgg_model_19.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_20 = vgg_model_20.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_21 = vgg_model_21.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_22 = vgg_model_22.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_23 = vgg_model_23.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_24 = vgg_model_24.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_25 = vgg_model_25.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_26 = vgg_model_26.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_pushi_area = vgg_model_pushi_area.VGG(img_size=opt.img_size, input_channel=1, num_class=14).cuda()
        net_16 = vgg_model_16.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_17 = vgg_model_17.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_18 = vgg_model_18.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_19 = vgg_model_19.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_20 = vgg_model_20.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_21 = vgg_model_21.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_22 = vgg_model_22.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_23 = vgg_model_23.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_24 = vgg_model_24.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_25 = vgg_model_25.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
        net_26 = vgg_model_26.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()

    # load model data
    net_pushi_area.load_state_dict(torch.load(opt.model_path_pushi_area))
    net_16.load_state_dict(torch.load(opt.model_path_16))
    net_17.load_state_dict(torch.load(opt.model_path_17))
    net_18.load_state_dict(torch.load(opt.model_path_18))
    net_19.load_state_dict(torch.load(opt.model_path_19))
    net_20.load_state_dict(torch.load(opt.model_path_20))
    net_21.load_state_dict(torch.load(opt.model_path_21))
    net_22.load_state_dict(torch.load(opt.model_path_22))
    net_23.load_state_dict(torch.load(opt.model_path_23))
    net_24.load_state_dict(torch.load(opt.model_path_24))
    net_25.load_state_dict(torch.load(opt.model_path_25))
    net_26.load_state_dict(torch.load(opt.model_path_26))
    net_pushi_area.eval()
    net_16.eval()
    net_17.eval()
    net_18.eval()
    net_19.eval()
    net_20.eval()
    net_21.eval()
    net_22.eval()
    net_23.eval()
    net_24.eval()
    net_25.eval()
    net_26.eval()

    # 使用 glob 模块匹配文件夹中的所有 png 文件
    for i in range(15, 29):
        csv_files = glob.glob(os.path.join(opt.img_path, "{}".format(i), '*.png'))  # 这个列表是跟文件夹倒序的
        num_batches = len(csv_files)  # 总共的组数
        for file in os.listdir(os.path.join(opt.img_path, "{}".format(i))):  # 从文件夹获取图片数据的文件名
            file_path = os.path.join(opt.img_path, "{}".format(i), file)  # 组合文件路径
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

            outputs_pushi_area = net_pushi_area(img)
            output_23 = net_23(img)
            # 获得面积
            x = int(float(file.split('_')[5]))  # 获得面积参数
            if x > 600:
                x = 0
            area_shu = float(int(pengzhangfenbu(x, 30))) / 80
            area_l = []
            for i in range(0, 200):
                area_l.append(area_shu)
            # print(area_l)
            area = torch.tensor(area_l)

            # 普适分类器结果
            # 在第二个维度上使用 torch.cat 进行拼接操作
            connect = torch.cat((outputs_pushi_area, area.unsqueeze(0)), dim=1)
            outputs_pushi_area = connect
            for fc in net_pushi_area.fc_list:  # 3 FC
                outputs_pushi_area = fc(outputs_pushi_area)

            percentage = torch.nn.functional.softmax(outputs_pushi_area, dim=1)[0] * 100  # 普适分类器各类别索引列表
            # print('predicted:', result)
            # 获取前三个最大的数据
            top_three_values = sorted(percentage, reverse=True)[:3]
            # 获取前三个最大数据的索引
            top_three_indices = sorted(range(len(percentage)), key=lambda i: percentage[i], reverse=True)[:3]

            # perc = percentage[int(indices)].item()  # 将最大的概率拿出赋值给perc
            result_1 = class_names[top_three_indices[0]]  # 将最大概率的数字标签拿到,字符型
            perc_1 = percentage[top_three_indices[0]]  # 将最大概率的拿到

            result_2 = class_names[top_three_indices[1]]  # 将第二概率的数字标签拿到
            perc_2 = percentage[top_three_indices[1]]  # 将第二的概率拿到
            result_3 = class_names[top_three_indices[2]]  # 将第三概率的数字标签拿到
            perc_3 = percentage[top_three_indices[2]]  # 将第三的概率拿到

            wenzilabel_1 = label_array[int(result_1) - 15]  # 概率最大的文字标签
            wenzilabel_2 = label_array[int(result_2) - 15]
            wenzilabel_3 = label_array[int(result_3) - 15]

            # cygdt_23分类器结果
            # 在第二个维度上使用 torch.cat 进行拼接操作
            connect = torch.cat((output_23, area.unsqueeze(0)), dim=1)
            outputs_23 = connect
            for fc in net_23.fc_list:  # 3 FC
                outputs_23 = fc(outputs_23)
            output_gailv_23, _ = torch.max(output_23, 1)  # cygdt_23的结果

            # 各个分类器结果
            output_16 = net_16(img)
            output_17 = net_17(img)
            output_18 = net_18(img)
            output_19 = net_19(img)
            output_20 = net_20(img)
            output_21 = net_21(img)
            output_22 = net_22(img)
            output_24 = net_24(img)
            output_25 = net_25(img)
            output_26 = net_26(img)
            output_gailv_16, _ = torch.max(output_16, 1)
            output_gailv_17, _ = torch.max(output_17, 1)
            output_gailv_18, _ = torch.max(output_18, 1)
            output_gailv_19, _ = torch.max(output_19, 1)
            output_gailv_20, _ = torch.max(output_20, 1)
            output_gailv_21, _ = torch.max(output_21, 1)
            output_gailv_22, _ = torch.max(output_22, 1)
            output_gailv_24, _ = torch.max(output_24, 1)
            output_gailv_25, _ = torch.max(output_25, 1)
            output_gailv_26, _ = torch.max(output_26, 1)

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
            fuheresult = []
            fuheresult[0] = perc_1
            fuheresult[1] = output_gailv_16
            fuheresult[2] = output_gailv_17
            fuheresult[3] = output_gailv_18
            fuheresult[4] = output_gailv_19
            fuheresult[5] = output_gailv_20
            fuheresult[6] = output_gailv_21
            fuheresult[7] = output_gailv_22
            fuheresult[8] = output_gailv_23
            fuheresult[9] = output_gailv_24
            fuheresult[10] = output_gailv_25
            fuheresult[11] = output_gailv_26
            fuheresult[12] = 0
            fuheresult[13] = 0
            print(file_path)
            print(wenzilabel_1)
            print(perc_1)
            print(("供液不足:{},气体影响:{}, 游动阀漏失:{}, 固定阀漏失:{}, 双阀漏失:{},上碰泵:{}, 下碰泵:{}, 抽油杆断脱:{}, 柱塞脱出工作筒:{}, 稠油:{}, 油井出砂:{}, "
                   "错误功图:{}, 管漏失:{}").format(fuheresult[1], fuheresult[2], fuheresult[3], fuheresult[4],
                                                    fuheresult[5],
                                                    fuheresult[6], fuheresult[7], fuheresult[8], fuheresult[9],
                                                    fuheresult[10], fuheresult[11], fuheresult[12],
                                                    fuheresult[13]))
            # 拆分文件名
            parts = file.split("_")
            # 修改第6个位置的部分
            parts.insert(6, ("供液不足:{:.2f},气体影响:{:.2f}, 游动阀漏失:{:.2f}, 固定阀漏失:{:.2f}, 双阀漏失:{:.2f},上碰泵:{:.2f}, 下碰泵:{:.2f}, "
                             "抽油杆断脱:{:.2f}, 柱塞脱出工作筒:{:.2f}, 稠油:{:.2f},"
                             "油井出砂:{},"
                             "错误功图:{}, 管漏失:{}").format(fuheresult[1], fuheresult[2], fuheresult[3], fuheresult[4],
                                                              fuheresult[5],
                                                              fuheresult[6], fuheresult[7], fuheresult[8], fuheresult[9],
                                                              fuheresult[10], fuheresult[11], fuheresult[12],
                                                              fuheresult[13]))

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
