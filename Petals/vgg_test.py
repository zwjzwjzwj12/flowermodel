# vgg_test.py
import os.path
import shutil
import time
import torch
import argparse
import vgg_model, vgg_dataset

from torch.utils.data import DataLoader

class_names = ["15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"]
label_array = ["工况正常", "供液不足", "气体影响", "游动阀漏失", "固定阀漏失", "双阀漏失",
               "上碰泵", "下碰泵", "抽油杆断脱", "柱塞脱出工作筒", "稠油", "油井出砂", "错误功图", "管漏失"]


def val(test_dataset):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    correct_zong = 0.0
    correct_erjianrong = 0.0
    # print(test_dataset.data_list)
    len_0 = test_dataset.__len__()
    step = 0
    label_8 = 0  # 统计有多少标签为8的
    true_8 = 0  # 统计多少8预测正确的
    for (images, labels, area) in test_dataset:
        if opt.gpu:
            images = images.cuda()
            labels = labels.cuda()
            area = area.cuda()
        outputs = net(images.unsqueeze(0))

        # 在第二个维度上使用 torch.cat 进行拼接操作
        connect = torch.cat((outputs, area.unsqueeze(0)), dim=1)
        outputs = connect
        for fc in net.fc_list:  # 3 FC
            outputs = fc(outputs)
        # _, preds = outputs.max(1)
        # 单标签判错：
        # if not preds.eq(torch.tensor([labels[0]]).to(outputs.device)):  # 如果第一标签与模型判断的标签不一致
        #     print(preds[0]+15)  # 模型的标签
        #     print(test_dataset.data_list[step])  # 判断错误的图片
        #     source_path = test_dataset.data_list[step]
        #     destination_path = "D:\\aboutjiqixuexi\\vgg19\\pushi\\judge_false"
        #     shutil.move(source_path, destination_path)  # 将盘错的图片剪切至 destination_path
        _, indices = torch.max(outputs, 1)
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100  # 各类别索引列表
        perc = percentage[int(indices)].item()  # 最大概率
        result = class_names[indices]  # 最大概率数字标签

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

        if labels[0] == 8:
            label_8 = label_8 + 1

        if not (int(result_1) - 15 == labels[0]
                or int(result_1) - 15 == labels[1]
                or int(result_1) - 15 == labels[2]
                or int(result_1) - 15 == labels[3]
                # or int(result_2) - 15 == labels[0]
                # or int(result_2) - 15 == labels[1]
                # or int(result_2) - 15 == labels[2]
                # or int(result_2) - 15 == labels[3]
        ):  # 如果4个标签与模型判断的前两个标签都不一致
            print(result_1)  # 模型的标签
            print(test_dataset.data_list[step])  # 判断错误的图片
            source_path = test_dataset.data_list[step]
            destination_path = "E:\\aboutjiqixuexi\\pushi_area\\judge_false_add.area_duibi_100_zaici_guanloushi_addgls_lunwentest"
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            parts = test_dataset.data_list[step].split("\\")[-1].split("_")
            parts.insert(6, "m{},{}+m{},{}".format(result_1, perc_1, result_2, perc_2))
            new_filename = "_".join(parts)
            des_path = os.path.join(destination_path, new_filename)
            shutil.copy(source_path, des_path)  # 将盘错的图片复制至 destination_path
        if int(result_1) - 15 == 8 and labels[0] == 8:  # 如果抽油杆断脱预测正确
            true_8 = true_8 + 1
        if int(result_1) - 15 != 8 and labels[0] == 8:  # 如果抽油杆断脱预测错误
            print(result_1)  # 模型的标签
            print(test_dataset.data_list[step])  # 抽油杆断脱图片
            source_path = test_dataset.data_list[step]
            destination_path = "E:\\aboutjiqixuexi\\pushi_area\\judge_true_8_add.area_100_zaici_addgls_lunwentest"
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            parts = test_dataset.data_list[step].split("\\")[-1].split("_")
            parts.insert(6, "m{},{}+m{},{}".format(result_1, perc_1, result_2, perc_2))
            new_filename = "_".join(parts)
            des_path = os.path.join(destination_path, new_filename)
            shutil.copy(source_path, des_path)  # 将盘错的图片复制至 destination_path
            # shutil.move(source_path, destination_path)  # 将盘错的图片剪切至 destination_path
        step = step + 1

        # 单个标签精度：
        # correct_zong += preds.eq(torch.tensor([labels[0]]).to(outputs.device)).sum()
        correct_zong += int(result_1) - 15 == labels[0]
        # # 包容性精度：
        # correct += preds.eq(torch.tensor([labels[0]]).to(outputs.device)).sum()
        # correct += preds.eq(torch.tensor([labels[1]]).to(outputs.device)).sum()
        # correct += preds.eq(torch.tensor([labels[2]]).to(outputs.device)).sum()
        # correct += preds.eq(torch.tensor([labels[3]]).to(outputs.device)).sum()
        # 包容性精度：第一标签包容性：
        correct += int(result_1) - 15 == labels[0]
        correct += int(result_1) - 15 == labels[1]
        correct += int(result_1) - 15 == labels[2]
        correct += int(result_1) - 15 == labels[3]
        # 两标签包容性精度：第一标签包容性：
        if int(result_1) - 15 == labels[0]:
            correct_erjianrong += 1
        if int(result_1) - 15 == labels[1]:
            correct_erjianrong += 1
        if int(result_1) - 15 == labels[2]:
            correct_erjianrong += 1
        if int(result_1) - 15 == labels[3]:
            correct_erjianrong += 1
        if (int(result_1) - 15 != labels[0]
                and int(result_1) - 15 != labels[1]
                and int(result_1) - 15 != labels[2]
                and int(result_1) - 15 != labels[3]):
            if int(result_2) - 15 == labels[0]:
                correct_erjianrong += 1
            if int(result_2) - 15 == labels[1]:
                correct_erjianrong += 1
            if int(result_2) - 15 == labels[2]:
                correct_erjianrong += 1
            if int(result_2) - 15 == labels[3]:
                correct_erjianrong += 1
        # 复合标签判错：
        # if not (preds.eq(torch.tensor([labels[0]]).to(outputs.device))
        #         | preds.eq(torch.tensor([labels[1]]).to(outputs.device))
        #         | preds.eq(torch.tensor([labels[2]]).to(outputs.device))
        #         | preds.eq(torch.tensor([labels[3]]).to(outputs.device))):  # 如果4个标签与模型判断的标签都不一致
        #     print(preds[0] + 15)  # 模型的标签
        #     print(test_dataset.data_list[step])  # 判断错误的图片
        #     source_path = test_dataset.data_list[step]
        #     destination_path = "D:\\aboutjiqixuexi\\vgg19\\pushi\\judge_false_3_checkpoint_jiaquansunshi_qingli23_pineilrxiajiang"
        #     if not os.path.exists(destination_path):
        #         os.makedirs(destination_path)
        #     parts = test_dataset.data_list[step].split("\\")[-1].split("_")
        #     parts.insert(6, "m{}".format(preds[0] + 15))
        #     new_filename = "_".join(parts)
        #     des_path = os.path.join(destination_path, new_filename)
        #     shutil.copy(source_path, des_path)  # 将盘错的图片复制至 destination_path
        #     # shutil.move(source_path, destination_path)  # 将盘错的图片剪切至 destination_path
        # step = step + 1
        #
        # # 单个标签精度：
        # # correct += preds.eq(torch.tensor([labels[0]]).to(outputs.device)).sum()
        #
        # # 包容性精度：
        # correct += preds.eq(torch.tensor([labels[0]]).to(outputs.device)).sum()
        # correct += preds.eq(torch.tensor([labels[1]]).to(outputs.device)).sum()
        # correct += preds.eq(torch.tensor([labels[2]]).to(outputs.device)).sum()
        # correct += preds.eq(torch.tensor([labels[3]]).to(outputs.device)).sum()

    print(true_8 / label_8)

    finish = time.time()

    test_acc = correct.float() / len(test_loader.dataset)
    zongjingdu = correct_zong.float() / len(test_loader.dataset)
    liangbrjingdu = correct_erjianrong / len(test_loader.dataset)
    print('总精度：{:.4f}, 单标签包容性精度：{:.4f},两标签包容性精度：{:.4f}, Time consumed:{:.2f}s, 抽油杆断脱acc：{:.4f}'.format(
        zongjingdu,
        test_acc,
        liangbrjingdu,
        finish - start,
        true_8 / label_8
    ))

    return test_loss, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str, default="test_dataset_genai_area_zuizhong",
                        help='images path')
    parser.add_argument('-model_path', type=str,
                        default="E:\\aboutjiqixuexi\\pushi_area\\checkpoint_area_200_addgls\\epoch_62_train_dataset_genai_area_zuizhong-best-acc_0.9784995317459106-jracc_0.9901646971702576-erjracc_1.0.pth",
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

    test_dataset = vgg_dataset.MyDataset("Test", opt.img_size, opt.img_path)
    test_loader = DataLoader(test_dataset, shuffle=True)

    val(test_dataset)
