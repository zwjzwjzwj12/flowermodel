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
    return 1 / (1 + np.exp(-x)) - 0.5


# 定义对下碰泵的膨胀分布函数
def pengzhangfenbu_22(x, center):
    qujian2 = [0, 0.5]
    q2len = qujian2[1] - qujian2[0]
    daqujian2 = [center, 1]
    dq2len = daqujian2[1] - daqujian2[0]

    if x > center:
        y = transformed_sigmoid(15*(x - center) / (dq2len / q2len)) * (dq2len / q2len) + center
        return y
    else:
        return x


# 定义对柱塞脱出工作筒的膨胀分布函数
def pengzhangfenbu_24(x, center):
    qujian2 = [0, 0.5]
    q2len = qujian2[1] - qujian2[0]
    daqujian2 = [center, 1]
    dq2len = daqujian2[1] - daqujian2[0]

    if x < center:
        y = transformed_sigmoid(10*(x - center) / (dq2len / q2len)) * (dq2len / q2len) + center
        return y
    else:
        return x
class Labelper:
    def __init__(self, my_float, my_string):
        self.per = my_float
        self.label = my_string


# 定义变换后的 sigmoid 函数
def transformed_sigmoid_area(x):
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
        y = transformed_sigmoid_area((x - center) / (dq1len / q1len)) * (dq1len / q1len) + center
        return y
    else:
        y = transformed_sigmoid_area((x - center) / (dq2len / q2len)) * (dq2len / q2len) + center
        return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str, default="E:\\aboutjiqixuexi\\zhmx_zong\\test_dataset_genai_area_zuizhong\\28\\26_28_26_40_40_40.00_管漏失_99.99644470214844_+_26_油井出砂_0.0033363578841090202_+_20_双阀漏失_6.178570038173348e-05_CTZ12.png",
                        help='images path')
    parser.add_argument('-new_img_path', type=str,
                        default="E:\\aboutjiqixuexi\\zhmx_zong\\makelabel_zh_zong_softmax_test",
                        help='new images path')
    parser.add_argument('-model_path_pushi_area', type=str,
                        default="./moxing_path\\epoch_62_train_dataset_genai_area_zuizhong-best-acc_0.9784995317459106-jracc_0.9901646971702576-erjracc_1.0.pth",
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

    if opt.gpu:
        net_pushi_area = vgg_model_pushi_area.VGG(img_size=opt.img_size, input_channel=1,
                                                  num_class=14).cuda()
    else:
        net_pushi_area = vgg_model_pushi_area.VGG(img_size=opt.img_size, input_channel=1,
                                                  num_class=14).cuda()
        # load model data
    net_pushi_area.load_state_dict(torch.load(opt.model_path_pushi_area))
    net_pushi_area.eval()
    # initialize vgg
    if opt.gpu:
        net_16 = vgg_model_16.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_16 = vgg_model_16.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    net_16.load_state_dict(torch.load(opt.model_path_16))
    net_16.eval()
    # initialize vgg
    if opt.gpu:
        net_17 = vgg_model_17.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_17 = vgg_model_17.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    net_17.load_state_dict(torch.load(opt.model_path_17))
    net_17.eval()
    # initialize vgg
    if opt.gpu:
        net_18 = vgg_model_18.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_18 = vgg_model_18.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    net_18.load_state_dict(torch.load(opt.model_path_18))
    net_18.eval()
    # initialize vgg
    if opt.gpu:
        net_19 = vgg_model_19.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_19 = vgg_model_19.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    net_19.load_state_dict(torch.load(opt.model_path_19))
    net_19.eval()
    # initialize vgg
    if opt.gpu:
        net_20 = vgg_model_20.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_20 = vgg_model_20.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    net_20.load_state_dict(torch.load(opt.model_path_20))
    net_20.eval()
    # initialize vgg
    if opt.gpu:
        net_21 = vgg_model_21.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_21 = vgg_model_21.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    net_21.load_state_dict(torch.load(opt.model_path_21))
    net_21.eval()
    # initialize vgg
    if opt.gpu:
        net_22 = vgg_model_22.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_22 = vgg_model_22.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    net_22.load_state_dict(torch.load(opt.model_path_22))
    net_22.eval()
    # initialize vgg
    if opt.gpu:
        net_23 = vgg_model_23.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_23 = vgg_model_23.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    net_23.load_state_dict(torch.load(opt.model_path_23))
    net_23.eval()
    # initialize vgg
    if opt.gpu:
        net_24 = vgg_model_24.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_24 = vgg_model_24.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    net_24.load_state_dict(torch.load(opt.model_path_24))
    net_24.eval()
    # initialize vgg
    if opt.gpu:
        net_25 = vgg_model_25.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_25 = vgg_model_25.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    net_25.load_state_dict(torch.load(opt.model_path_25))
    net_25.eval()
    # initialize vgg
    if opt.gpu:
        net_26 = vgg_model_26.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net_26 = vgg_model_26.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    net_26.load_state_dict(torch.load(opt.model_path_26))
    net_26.eval()
    # 打开图片
    file_path = opt.img_path
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
    # initialize vgg
    outputs_pushi_area = net_pushi_area(img)

    # 获得面积
    file = os.path.basename(opt.img_path)
    x = int(float(file.split('_')[5]))  # 获得面积参数
    if x > 600:
        x = 0
    area_shu = float(int(pengzhangfenbu(x, 30))) / 80
    area_l = []
    for a in range(0, 200):
        area_l.append(area_shu)
    # print(area_l)
    area = torch.tensor(area_l)

    # 普适分类器结果
    # 在第二个维度上使用 torch.cat 进行拼接操作
    connect = torch.cat((outputs_pushi_area, area.unsqueeze(0).to(outputs_pushi_area.device)), dim=1)
    outputs_pushi_area = connect
    for fc in net_pushi_area.fc_list:  # 3 FC
        outputs_pushi_area = fc(outputs_pushi_area)

    percentage = torch.nn.functional.softmax(outputs_pushi_area, dim=1)[0]  # 普适分类器各类别索引列表
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
    # 各个分类器的输出
    output_16 = net_16(img)

    output_17 = net_17(img)

    output_18 = net_18(img)

    output_19 = net_19(img)

    output_20 = net_20(img)

    output_21 = net_21(img)

    output_22 = net_22(img)

    output_23 = net_23(img)
    # 在第二个维度上使用 torch.cat 进行拼接操作
    connect = torch.cat((output_23, area.unsqueeze(0).to(outputs_pushi_area.device)), dim=1)
    outputs_23 = connect
    for fc in net_23.fc_list:  # 3 FC
        outputs_23 = fc(outputs_23)
    output_gailv_23, _ = torch.max(outputs_23, 1)  # cygdt_23的结果

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
    fuheresult = [0, float(output_gailv_16), float(output_gailv_17), float(output_gailv_18),
                  float(output_gailv_19), float(output_gailv_20),
                  float(output_gailv_21), pengzhangfenbu_22(float(output_gailv_22), 0.5), float(output_gailv_23),
                  pengzhangfenbu_24(float(output_gailv_24), 0.5), float(output_gailv_25),
                  float(output_gailv_26), 0, 0]

    print(wenzilabel_1)
    print(perc_1)

    # 创建存放最终复合工况的列表
    fuhe_result = []

    # 先决定第一名，并将其加入列表
    # 15,27,28，是第一
    if int(result_1) == 15 or int(result_1) == 27 or int(result_1) == 28:
        obj = Labelper(float(perc_1), label_array[int(result_1) - 15])
        fuhe_result.append(obj)
        print(fuhe_result[0].per)
        print(fuhe_result[0].label)
        # 创建长度为14的数组，并向其中添加类对象
        for b in range(0, 14):
            if b == int(result_1) - 15:
                continue
            obj_0 = Labelper(fuheresult[b], label_array[b])
            fuhe_result.append(obj_0)
        # 打印这个结果表格
        print("第一种情况，第一标签为15，,27,28打印表格")
        for c in range(0, 14):
            print(fuhe_result[c].per)
            print(fuhe_result[c].label)

        # 指定排序范围，对第2位到第5位进行排序
        start_index = 1  # 范围起始位置的索引（从0开始）
        end_index = 14  # 范围结束位置的索引（不包含）

        # 选择要排序的子数组
        sub_array = fuhe_result[start_index:end_index]

        # 对子数组进行排序
        sub_array.sort(key=lambda obj: obj.per, reverse=True)

        # 将排序后的子数组放回原数组的对应位置
        fuhe_result[start_index:end_index] = sub_array
        print("排序后，第一标签为15，,27,28打印排序后表格")
        for c in range(0, 14):
            print(fuhe_result[c].per)
            print(fuhe_result[c].label)
        # 保证第一名是最大的
        c = 1
        while c <= 14:
            if fuhe_result[c].per > fuhe_result[c - 1].per:
                temp = fuhe_result[c - 1].per
                fuhe_result[c].per = temp-0.001
                print(fuhe_result[c-1].per)
                print(fuhe_result[c - 1].per)
            else:
                break
            c = c + 1
        print("保证第一名最大后，排序后，第一标签为15，,27,28打印排序后表格")
        for c in range(0, 14):
            print(fuhe_result[c].per)
            print(fuhe_result[c].label)
        # softmax处理
        he = 0.0
        for p in fuhe_result:
            he = he + p.per
        for k in range(0, 14):
            fuhe_result[k].per = fuhe_result[k].per / he
        print("soft之后，排序后，第一标签为15，,27,28打印排序后表格")
        for c in range(0, 14):
            print(fuhe_result[c].per)
            print(fuhe_result[c].label)
    # 15,27,28，不是第一,16或者17是第一
    if int(result_1) == 17 or int(result_1) == 16:
        if fuheresult[2] > 0.5:  # 相信气体影响的判断,是气体影响就将气体影响排第一
            obj = Labelper(fuheresult[2], label_array[2])
            fuhe_result.append(obj)
            for l in range(0, 14):
                if l == 2:
                    continue
                obj_0 = Labelper(fuheresult[l], label_array[l])
                fuhe_result.append(obj_0)
            # 指定排序范围，对第2位到第5位进行排序
            start_index = 1  # 范围起始位置的索引（从0开始）
            end_index = 14  # 范围结束位置的索引（不包含）

            # 选择要排序的子数组
            sub_array = fuhe_result[start_index:end_index]

            # 对子数组进行排序
            sub_array.sort(key=lambda obj: obj.per, reverse=True)

            # 将排序后的子数组放回原数组的对应位置
            fuhe_result[start_index:end_index] = sub_array
            # 保证第一名是最大的
            c = 1
            while c <= 14:
                if fuhe_result[c].per > fuhe_result[c - 1].per:
                    fuhe_result[c].per = fuhe_result[c - 1].per - 0.001
                else:
                    break
                c = c + 1
            # softmax处理
            he = 0.0
            for p in fuhe_result:
                he = he + p.per
            for k in range(0, 14):
                fuhe_result[k].per = fuhe_result[k].per / he
        else:
            obj = Labelper(fuheresult[1], label_array[1])
            fuhe_result.append(obj)
            for l in range(0, 14):
                if l == 1:
                    continue
                obj_0 = Labelper(fuheresult[l], label_array[l])
                fuhe_result.append(obj_0)
            # 指定排序范围，对第2位到第5位进行排序
            start_index = 1  # 范围起始位置的索引（从0开始）
            end_index = 14  # 范围结束位置的索引（不包含）

            # 选择要排序的子数组
            sub_array = fuhe_result[start_index:end_index]

            # 对子数组进行排序
            sub_array.sort(key=lambda obj: obj.per, reverse=True)

            # 将排序后的子数组放回原数组的对应位置
            fuhe_result[start_index:end_index] = sub_array
            # 保证第一名是最大的
            c = 1
            while c <= 13:
                if fuhe_result[c].per > fuhe_result[c - 1].per:
                    fuhe_result[c].per = fuhe_result[c - 1].per - 0.001
                else:
                    break
                c = c + 1
            # softmax处理
            he = 0.0
            for p in fuhe_result:
                he = he + p.per
            for k in range(0, 14):
                fuhe_result[k].per = fuhe_result[k].per / he
    # 18,19,20,21,22,23,24,25,26是第一
    if int(result_1) == 18 or int(result_1) == 19 or int(result_1) == 20 or int(result_1) == 21 or int(
            result_1) == 22 or int(result_1) == 23 or int(result_1) == 24 or int(result_1) == 25 or int(
        result_1) == 26:
        qianliangming = []
        qianliangming.append(fuheresult[int(result_1) - 15])
        if int(result_2) == 26 or int(result_2) == 25:
            qianliangming.append(0)
        else:
            qianliangming.append(fuheresult[int(result_2) - 15])

        if qianliangming[0] > qianliangming[1]:
            obj = Labelper(fuheresult[int(result_1) - 15], label_array[int(result_1) - 15])
            fuhe_result.append(obj)
            for l in range(0, 14):
                if l == int(result_1) - 15:
                    continue
                obj_0 = Labelper(fuheresult[l], label_array[l])
                fuhe_result.append(obj_0)
            # 指定排序范围，对第2位到第5位进行排序
            start_index = 1  # 范围起始位置的索引（从0开始）
            end_index = 14  # 范围结束位置的索引（不包含）

            # 选择要排序的子数组
            sub_array = fuhe_result[start_index:end_index]

            # 对子数组进行排序
            sub_array.sort(key=lambda obj: obj.per, reverse=True)

            # 将排序后的子数组放回原数组的对应位置
            fuhe_result[start_index:end_index] = sub_array
            # 保证第一名是最大的
            c = 1
            while c <= 14:
                if fuhe_result[c].per > fuhe_result[c - 1].per:
                    fuhe_result[c].per = fuhe_result[c - 1].per - 0.001
                else:
                    break
                c = c + 1
            # softmax处理
            he = 0.0
            for p in fuhe_result:
                he = he + p.per
            for k in range(0, 14):
                fuhe_result[k].per = fuhe_result[k].per / he
        else:
            obj = Labelper(fuheresult[int(result_2) - 15], label_array[int(result_2) - 15])
            fuhe_result.append(obj)
            for l in range(0, 14):
                if l == int(result_2) - 15:
                    continue
                obj_0 = Labelper(fuheresult[l], label_array[l])
                fuhe_result.append(obj_0)
            # 指定排序范围，对第2位到第5位进行排序
            start_index = 1  # 范围起始位置的索引（从0开始）
            end_index = 14  # 范围结束位置的索引（不包含）

            # 选择要排序的子数组
            sub_array = fuhe_result[start_index:end_index]

            # 对子数组进行排序
            sub_array.sort(key=lambda obj: obj.per, reverse=True)

            # 将排序后的子数组放回原数组的对应位置
            fuhe_result[start_index:end_index] = sub_array
            # 保证第一名是最大的
            c = 1
            while c <= 14:
                if fuhe_result[c].per > fuhe_result[c - 1].per:
                    fuhe_result[c].per = fuhe_result[c - 1].per - 0.001
                else:
                    break
                c = c + 1
            # softmax处理
            he = 0.0
            for p in fuhe_result:
                he = he + p.per
            for k in range(0, 14):
                fuhe_result[k].per = fuhe_result[k].per / he

    # 拆分文件名
    parts = file.split("_")
    # 修改第6个位置的部分
    print(6, ("{}{:.1f}-{}{:.1f}-{}{:.1f}-{}{:.1f}-{}{:.1f}-{}{:.1f}-{}{:.1f}-{}{:.1f}-"
              "{}{:.1f}-{}{:.1f}-{}{:.1f}-"
              "{}{:.1f}-"
              "{}{:.1f}-{}{:.1f}").format(fuhe_result[0].label, fuhe_result[0].per * 100,
                                          fuhe_result[1].label, fuhe_result[1].per * 100,
                                          fuhe_result[2].label, fuhe_result[2].per * 100,
                                          fuhe_result[3].label, fuhe_result[3].per * 100,
                                          fuhe_result[4].label, fuhe_result[4].per * 100,
                                          fuhe_result[5].label, fuhe_result[5].per * 100,
                                          fuhe_result[6].label, fuhe_result[6].per * 100,
                                          fuhe_result[7].label, fuhe_result[7].per * 100,
                                          fuhe_result[8].label, fuhe_result[8].per * 100,
                                          fuhe_result[9].label, fuhe_result[9].per * 100,
                                          fuhe_result[10].label, fuhe_result[10].per * 100,
                                          fuhe_result[11].label, fuhe_result[11].per * 100,
                                          fuhe_result[12].label, fuhe_result[12].per * 100,
                                          fuhe_result[13].label, fuhe_result[13].per * 100,
                                          ))

    # 修改第6个位置的部分
    parts.insert(6, ("{}{:.1f}-{}{:.1f}-{}{:.1f}-{}{:.1f}-{}{:.1f}-{}{:.1f}-{}{:.1f}-{}{:.1f}-"
                     "{}{:.1f}-{}{:.1f}-{}{:.1f}-"
                     "{}{:.1f}-"
                     "{}{:.1f}-{}{:.1f}").format(fuhe_result[0].label, fuhe_result[0].per * 100,
                                                 fuhe_result[1].label, fuhe_result[1].per * 100,
                                                 fuhe_result[2].label, fuhe_result[2].per * 100,
                                                 fuhe_result[3].label, fuhe_result[3].per * 100,
                                                 fuhe_result[4].label, fuhe_result[4].per * 100,
                                                 fuhe_result[5].label, fuhe_result[5].per * 100,
                                                 fuhe_result[6].label, fuhe_result[6].per * 100,
                                                 fuhe_result[7].label, fuhe_result[7].per * 100,
                                                 fuhe_result[8].label, fuhe_result[8].per * 100,
                                                 fuhe_result[9].label, fuhe_result[9].per * 100,
                                                 fuhe_result[10].label, fuhe_result[10].per * 100,
                                                 fuhe_result[11].label, fuhe_result[11].per * 100,
                                                 fuhe_result[12].label, fuhe_result[12].per * 100,
                                                 fuhe_result[13].label, fuhe_result[13].per * 100,
                                                 ))

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
