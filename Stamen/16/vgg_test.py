# vgg_test.py
import os.path
import shutil
import time
import torch
import argparse
import vgg_model, vgg_dataset

from torch.utils.data import DataLoader

class_names = ["15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27"]
label_array = ["工况正常", "供液不足", "气体影响", "游动阀漏失", "固定阀漏失", "双阀漏失",
               "上碰泵", "下碰泵", "抽油杆断脱", "柱塞脱出工作筒", "稠油", "油井出砂", "错误功图"]


def val(test_dataset):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    tp = 0
    tpfp = 0.0
    tpfn = 0.0
    # print(test_dataset.data_list)
    len_0 = test_dataset.__len__()
    step = 0
    for (images, labels) in test_dataset:
        if opt.gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images.unsqueeze(0))

        # _, preds = outputs.max(1)
        # 单标签判错：
        # if not preds.eq(torch.tensor([labels[0]]).to(outputs.device)):  # 如果第一标签与模型判断的标签不一致
        #     print(preds[0]+15)  # 模型的标签
        #     print(test_dataset.data_list[step])  # 判断错误的图片
        #     source_path = test_dataset.data_list[step]
        #     destination_path = "D:\\aboutjiqixuexi\\vgg19\\pushi\\judge_false"
        #     shutil.move(source_path, destination_path)  # 将盘错的图片剪切至 destination_path

        if (outputs >= 0.5 and labels != 1.0) or (outputs < 0.5 and labels != 0.0):
            # 如果预测错误
            print(labels)  # 模型的标签
            print(test_dataset.data_list[step])  # 判断错误的图片
            source_path = test_dataset.data_list[step]
            destination_path = "E:\\aboutjiqixuexi\\gybz_16\\judge_false_add.area_gybz_16_sig(xia_chu120_shang_chu25)"
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            parts = (source_path.split("\\")[-1]).split("_")
            print(parts)  # 模型的标签
            label = 0
            if outputs >= 0.5:
                label = 1
            else:
                label = 0
            gailv = float(outputs[:, 0])
            parts.insert(6, "m{}+供液不足的概率{}".format(label, gailv))
            new_filename = "_".join(parts)
            print(new_filename)
            des_path = os.path.join(destination_path, new_filename)
            shutil.copy(source_path, des_path)  # 将盘错的图片复制至 des_path
            # shutil.move(source_path, des_path)  # 将判断错的图片剪切至 des_path
        step = step + 1

        # 精度
        if outputs >= 0.5 and labels == 1.0:  # 正样本预测正确,真正的正例
            correct = correct + 1.0
            tp = tp+1.0
        if outputs < 0.5 and labels == 0.0:  # 负样本预测正确
            correct = correct + 1.0
        if outputs >= 0.5:  # 我认为正样本
            tpfp = tpfp + 1.0
        if labels == 1.0:  # 实际的正样本总数
            tpfn = tpfn+1.0

    finish = time.time()

    test_acc = correct / len(test_loader.dataset)
    p = tp/tpfp
    r = tp/tpfn
    print('Test set: Accuracy: {:.4f}, p:{:.2f},r:{:.2f},Time consumed:{:.2f}Zs,'.format(
        test_acc,
        p, r,
        finish - start,
    ))

    return test_loss, test_acc, p, r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str, default="test_dataset_genai_area_zuizhong",
                        help='images path')
    parser.add_argument('-model_path', type=str,
                        default="E:\\aboutjiqixuexi\\gybz_16\\checkpoint_wei_sig(shang_chu25_xia_chu120)\\epoch_60_train_dataset_genai_area_zuizhong-last-acc_0.9961945031712474.pth",
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

    test_dataset = vgg_dataset.MyDataset("Test", opt.img_size, opt.img_path)
    test_loader = DataLoader(test_dataset, shuffle=True)

    val(test_dataset)
