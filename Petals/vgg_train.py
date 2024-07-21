# vgg_train.py

import argparse
import logging
import os
import time

import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import vgg_dataset
import vgg_model

class_names = ["15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"]
label_array = ["工况正常", "供液不足", "气体影响", "游动阀漏失", "固定阀漏失", "双阀漏失",
               "上碰泵", "下碰泵", "抽油杆断脱", "柱塞脱出工作筒", "稠油", "油井出砂", "错误功图", "管漏失"]

logger = logging.getLogger(__name__)
# logging：这是 Python 内置的日志记录模块。通过使用这个模块，我们可以在程序中记录日志消息，以便在后续的调试和分析中使用。
# getLogger(__name__)：logging.getLogger() 是 logging 模块中的一个函数，用于获取一个 logger 对象。
# __name__ 是一个预定义的变量，表示当前模块的名称。通常，我们在一个模块中创建一个 logger 对象，并使用该对象记录与该模块相关的日志消息。


# train each epoch  训练训练集模块
train_time = 0


def train(epoch, pbar):
    start = time.time()
    net.train()
    # train()：这是神经网络模型对象的一个方法。在深度学习中，神经网络模型通常有两种模式：训练模式和评估模式。train() 方法用于将模型设置为训练模式。
    #
    # 当将神经网络模型设置为训练模式时，它会启用一些特定的行为，例如启用 Dropout 层的随机失活、启用 Batch Normalization 层的批次统计更新等。
    # 此外，训练模式还会影响模型的计算图、梯度计算和参数更新等过程。
    #
    # 通过调用 net.train() 方法，我们可以确保在进行训练过程时，神经网络模型处于正确的模式下，以便进行模型训练和参数更新。
    # 这在深度学习中非常常见，通常在每个训练迭代或训练循环的开始处调用 train() 方法来确保模型处于训练模式。

    #  start batch
    for step, (images, labels, area) in pbar:
        torch.set_printoptions(threshold=np.inf)
        if opt.gpu:
            images, labels, area = images.cuda(), labels.cuda(), area.cuda()
        optimizer.zero_grad()  # 清除之前批次的梯度信息。

        outputs = net(images)
        # 在第二个维度上使用 torch.cat 进行拼接操作
        connect = torch.cat((outputs, area), dim=1)
        outputs = connect
        torch.set_printoptions(threshold=np.inf)
        for fc in net.fc_list:        # 3 FC
            outputs = fc(outputs)
        # outputs = net(combined_tensor)
        column = torch.tensor([labels[:, 0]]).to(outputs.device)  # 真实标签
        # 单独的：代表全部，获取第2，3行和第2，3列的子张量，sub_tensor = tensor[1:3, 1:3]，两行两列
        # 子张量转化为张量：sub_tensor = tensor[1:3, 1:3].clone()
        # .clone() 方法用于创建一个原始张量的独立副本。这意味着它会复制原始张量的数据和形状，并返回一个新的张量对象，与原始张量完全独立。
        if [labels[:, 0]] == 8:  # 加权第8个类的损失权重
            loss = loss_function(outputs, column) * 15
        elif labels[:, 0] == 10:
            loss = loss_function(outputs, column) * 5
        else:
            loss = loss_function(outputs, column)
        loss.backward()
        optimizer.step()  # optimizer.step()：根据计算得到的梯度更新模型的参数，执行一步优化。
        # .step() 方法的作用就是根据计算得到的梯度更新模型的参数。它根据优化算法（如随机梯度下降）的规则，调整参数的值以最小化损失函数。
        # train_scheduler.step()  # train_scheduler.step()：更新训练学习率调度器（scheduler）的状态。
        # # 学习率调度器中的 step() 方法可以感知当前的 epoch 是多少。
        # # 学习率调度器内部会维护一个计数器，记录已经经过的训练周期数。在每次调用 step() 方法时，计数器会自增1，并与里程碑列表中的周期数进行比较。
        # # 如果计数器的值与里程碑列表中的某个周期数相匹配，学习率调度器会触发学习率的调整。
        # # print(train_scheduler.get_lr()[-1])
        # 判断是否到达衰减的批次位置
        # print(optimizer.param_groups[0]['lr'])
        if epoch <= opt.milestones[0] * opt.epochs:
            if step == int(nb * 0.5):
                optimizer.param_groups[0]['lr'] *= 0.8
            elif step == int(nb * 0.8):
                optimizer.param_groups[0]['lr'] *= 0.5
            elif step == int(nb * 0.9):
                optimizer.param_groups[0]['lr'] *= 0.2
        # train_scheduler.step()  # train_scheduler.step()：更新训练学习率调度器（scheduler）的状态。
        # 学习率调度器中的 step() 方法可以感知当前的 epoch 是多少。
        # 学习率调度器内部会维护一个计数器，记录已经经过的训练周期数。在每次调用 step() 方法时，计数器会自增1，并与里程碑列表中的周期数进行比较。
        # 如果计数器的值与里程碑列表中的某个周期数相匹配，学习率调度器会触发学习率的调整。
        # print(train_scheduler.get_lr())
        finish = time.time()
        s = ('epoch: %d\t loss: %10s\t lr: %10f\t' % (epoch, loss.item(), optimizer.param_groups[0]['lr']))
        # s = ('epoch: %d\t loss: %10s\t lr: %6f\t' % (epoch, loss.item(), train_scheduler.get_last_lr()[-1]))：
        # 将当前 epoch 的损失值和学习率格式化为字符串 s。
        # loss.item() 方法返回一个 Python 浮点数，表示损失张量中的数值。通过调用 item() 方法，
        # 我们可以将损失值从张量中提取出来，以便进行打印、记录或其他操作。
        # 需要注意的是，loss.item() 方法只能在标量张量上调用，即张量只包含一个数值。如果尝试在包含多个元素的张量上调用 item() 方法，将会引发错误。
        # 获取训练调度器（scheduler）的最后一个学习率（learning rate）的值
        pbar.set_description(s)
        global train_time
        train_time = finish - start
        # print(finish-start)
        # 将格式化的字符串 s 设置为进度条 pbar 的描述，用于显示当前训练进度和相关信息。

    # end batch


def val():
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    correct_jianrong = 0.0
    correct_erjianrong = 0.0
    for (images, labels, area) in test_loader:
        if opt.gpu:
            images = images.cuda()
            labels = labels.cuda()
            area = area.cuda()

        outputs = net(images)

        # 在第二个维度上使用 torch.cat 进行拼接操作
        connect = torch.cat((outputs, area), dim=1)
        outputs = connect
        for fc in net.fc_list:  # 3 FC
            outputs = fc(outputs)
        loss = loss_function(outputs, torch.tensor([labels[:, 0]]).to(outputs.device))
        test_loss += loss.item()
        _, preds = outputs.max(1)
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
        # 这段代码的作用是从模型的输出 outputs 中提取预测结果。通过使用 outputs.max(1)，
        # 我们获取了每个样本在各个类别上的最大值和对应的索引。通过将最大值的索引赋值给 preds 变量，我们得到了模型预测的类别标签。
        # "_"这是一个占位符变量，通常用于接收不需要使用的值。在这里，我们使用 _ 来接收最大值，但不使用它。
        # outputs：这是模型的输出结果，通常是一个张量（tensor），其中包含了每个样本在各个类别上的得分或概率。
        # outputs.max(1)：这是对 outputs 进行的操作，使用 max() 函数来获取每个样本在各个类别上的最大值，并返回这些最大值和对应的索引。
        correct += preds.eq(torch.tensor([labels[:, 0]]).to(outputs.device)).sum()
        correct_jianrong += preds.eq(torch.tensor([labels[:, 0]]).to(outputs.device)).sum()
        correct_jianrong += preds.eq(torch.tensor([labels[:, 1]]).to(outputs.device)).sum()
        correct_jianrong += preds.eq(torch.tensor([labels[:, 2]]).to(outputs.device)).sum()
        correct_jianrong += preds.eq(torch.tensor([labels[:, 3]]).to(outputs.device)).sum()
        if int(result_1) - 15 == labels[:, 0].item():
            correct_erjianrong += 1
        if int(result_1) - 15 == labels[:, 1].item():
            correct_erjianrong += 1
        if int(result_1) - 15 == labels[:, 2].item():
            correct_erjianrong += 1
        if int(result_1) - 15 == labels[:, 3].item():
            correct_erjianrong += 1
        if (int(result_1) - 15 != labels[:, 0].item()
                and int(result_1) - 15 != labels[:, 1].item()
                and int(result_1) - 15 != labels[:, 2].item()
                and int(result_1) - 15 != labels[:, 3].item()):
            if int(result_2) - 15 == labels[:, 0].item():
                correct_erjianrong += 1
            if int(result_2) - 15 == labels[:, 1].item():
                correct_erjianrong += 1
            if int(result_2) - 15 == labels[:, 2].item():
                correct_erjianrong += 1
            if int(result_2) - 15 == labels[:, 3].item():
                correct_erjianrong += 1
    finish = time.time()

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct.float() / len(test_loader.dataset)
    test_acc_jianrong = correct_jianrong.float() / len(test_loader.dataset)
    test_acc_erjianrong = correct_erjianrong / len(test_loader.dataset)
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Janrong_Accuracy: {:.4f},ErJanrong_Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            test_loss,
            test_acc,
            test_acc_jianrong,
            test_acc_erjianrong,
            finish - start
        ))

    return test_loss, test_acc, test_acc_jianrong, test_acc_erjianrong


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_path', type=str, default="train_dataset_genai_area_zuizhong", help='train images dir')
    parser.add_argument('-test_path', type=str, default="train_dataset_genai_area_zuizhong", help='test images dir')
    parser.add_argument('-img_size', type=int, default=224, help='the size of image, mutiple of 32')
    parser.add_argument('-num_class', type=int, default=14, help='the number of class')
    parser.add_argument('-checkpoint_path', type=str, default="checkpoint", help='path to save model')
    parser.add_argument('-batch_size', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-epochs', type=int, default=70, help='epochs')
    parser.add_argument('-milestones', type=float, default=[0.4, 0.6, 0.8], help='milestones')
    parser.add_argument('-gpu', default=True, help='use gpu or not')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate of optimizer')
    parser.add_argument('-tensorboard', default=True, help='use tensorboard or not')

    opt = parser.parse_args()
    # train_path = opt.train_path.split('/')[0]
    # initialize vgg
    if opt.gpu:
        net = vgg_model.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class).cuda()
    else:
        net = vgg_model.VGG(img_size=opt.img_size, input_channel=1, num_class=opt.num_class)

    # load data
    train_dataset = vgg_dataset.MyDataset("Train", opt.img_size, opt.train_path)
    test_dataset = vgg_dataset.MyDataset("Test", opt.img_size, opt.test_path)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=2, shuffle=True)
    # num_workers: 这是一个整数，表示用于数据加载的子进程数量。通过使用多个子进程，
    # 可以加速数据加载过程。一般建议将 num_workers 设置为大于 0 的整数，具体取决于系统资源和数据集大小。
    # Reference https://blog.csdn.net/zw__chen/article/details/82806900
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=2, shuffle=True)
    nb = len(train_dataset)

    # Optimzer
    loss_function = torch.nn.CrossEntropyLoss()  # 这行代码创建了一个交叉熵损失函数的实例，用于计算模型输出与真实标签之间的损失。
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    # net.parameters() 返回模型中所有可训练参数的迭代器，这些参数将在优化过程中进行更新。lr=opt.lr 指定了学习率，
    # momentum=0.9 指定了动量（momentum）参数，weight_decay=5e-4 指定了权重衰减（weight decay）参数。
    train_scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[x * opt.epochs for x in opt.milestones],
                                             gamma=0.1))
    # 这行代码创建了一个多步学习率调度器（scheduler）的实例，用于动态调整学习率。
    # optimizer 是要进行学习率调度的优化器实例，milestones 是一个列表或迭代器，用于指定在哪些 epoch 上更新学习率，gamma 是学习率调整的乘法因子。
    # learning rate decay
    # milestones的参数应该是一个列表
    houzhui = "area_200_addgls"
    # checkpoint  创建和管理模型的检查点（checkpoint）
    if not os.path.exists("{}_{}".format(opt.checkpoint_path, houzhui)):
        # 这行代码检查指定的检查点路径 opt.checkpoint_path 是否存在。
        # 如果路径不存在，则使用 os.makedirs() 创建该路径。这是为了确保检查点路径在保存之前已经存在。
        os.makedirs("{}_{}".format(opt.checkpoint_path, houzhui))
        # checkpoint_path是path to save model
    checkpoint_path = (
        os.path.join("{}_{}".format(opt.checkpoint_path, houzhui),
                     'epoch_{epoch}_{train_path}-{type}-acc_{acc}-jracc_{jracc}-erjracc_{erjracc}.pth'))
    # {epoch}：表示当前的训练 epoch（轮次）。
    # {type}：表示检查点的类型，可以是训练集（train）或验证集（val）等标识符。
    # {acc}：表示准确率（accuracy）或其他性能指标的数值。

    # tensorboard
    if opt.tensorboard:
        writer = SummaryWriter(log_dir="quxian_jiaquansunshi_qingli23_pineilrxiajiang_{}".format(houzhui))

    # Start train
    best_acc = 0.0
    for epoch in range(1, opt.epochs + 1):
        if epoch <= opt.milestones[0] * opt.epochs:
            optimizer.param_groups[0]['lr'] = opt.lr
        elif opt.milestones[1] * opt.epochs >= epoch > opt.milestones[0] * opt.epochs:
            optimizer.param_groups[0]['lr'] = opt.lr * 0.1
        else:
            optimizer.param_groups[0]['lr'] = opt.lr * 0.01
        pbar = tqdm(enumerate(train_loader), total=int(nb / opt.batch_size))  # process_bar
        train(epoch, pbar)  # train 1 epoch
        # print(train_scheduler.get_last_lr()[-1])
        # train_scheduler.step()  # train_scheduler.step()：更新训练学习率调度器（scheduler）的状态。
        print("此轮用时{}s".format(train_time))
        loss, acc, jracc, erjracc = val()  # valuation
        # tqdm 是一个 Python 库，用于创建进度条，以提供可视化的迭代进度信息。它的参数和效果如下：
        # tqdm(iterable, desc=None, total=None, leave=True, ncols=None, dynamic_ncols=False, mininterval=0.1, ...)
        # iterable：必需参数，表示要迭代的对象，例如列表、迭代器或生成器。
        # desc：可选参数，表示进度条的描述文本。默认为 None。
        # total：可选参数，表示总的迭代次数。当提供了该参数时，进度条将显示当前的进度百分比。
        # leave：可选参数，控制进度条完成后是否保留在终端上。默认为 True，即保留进度条。
        # ncols：可选参数，表示进度条的总宽度（以字符数为单位）。默认根据终端的宽度自动调整。
        # dynamic_ncols：可选参数，如果设置为 True，则动态调整进度条的宽度以适应终端大小的变化。默认为 False。
        # mininterval：可选参数，表示进度条更新的最小时间间隔（以秒为单位）。默认为 0.1 秒。
        # 除了上述列出的参数之外，tqdm 还提供了许多其他参数和选项，用于自定义进度条的外观和行为。例如，可以设置进度条的样式、动画效果、显示的单位等。
        # 效果：
        # 当使用 tqdm 包装一个迭代对象时，它会在终端上显示一个进度条，并且会在每次迭代时更新进度条的状态。
        # 进度条显示当前的进度百分比、已经经过的时间、预计剩余时间等信息，以及可选的描述文本。这样，我们可以通过可视化的方式实时查看迭代的进度，从而更直观地了解任务的执行情况。
        # 使用了 tqdm 库来创建一个进度条（progress bar），用于可视化迭代过程中的进度。让我逐步解释这段代码的含义：
        # enumerate(train_loader)：train_loader 是一个数据加载器（data loader），用于按批次加载训练数据。
        # enumerate() 函数用于在每个迭代步骤中返回一个索引和对应的数据批次。这个步骤将迭代器和索引组合在一起，以便在后续的循环中可以同时获取迭代的索引和数据。
        # nb 是总样本数，opt.batch_size 是批次大小。通过将总样本数除以批次大小，可以得到迭代的总步数。
        # 将这个值作为参数传递给 tqdm 的 total 参数，用于指定总的迭代次数。

        if opt.tensorboard:
            writer.add_scalar('Test/Average loss', loss, epoch)
            # add_scalar()：这是写入器对象的方法，用于将标量（scalar）数据写入到记录文件中。标量数据表示只有一个数值的数据，例如损失、准确率等。
            writer.add_scalar('Test/Accuracy', acc, epoch)
            writer.add_scalar('Test/jianrong_Accuracy', jracc, epoch)
            writer.add_scalar('Test/erjianrong_Accuracy', erjracc, epoch)

        if epoch > opt.epochs * opt.milestones[0] and best_acc < acc:
            weights_path = checkpoint_path.format(epoch=epoch, train_path=opt.train_path, type='best', acc=acc,
                                                  jracc=jracc, erjracc=erjracc)
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue
        # 这段代码是一个条件语句块，用于在满足一定条件时保存模型的权重（weights）文件，并更新最佳准确率（best_acc）的值。
        # epoch：这是当前训练的轮数。
        # opt.epochs：这是一个选项参数，表示总的训练轮数。
        # opt.milestones[1]：这是一个选项参数，是一个列表（或其他可迭代对象），其中包含里程碑轮数的索引。在这里，opt.milestones[1] 表示第二个里程碑轮数。
        # best_acc：这是当前的最佳准确率值。
        # acc：这是当前的准确率值。
        # 代码的条件语句判断了两个条件是否同时满足：
        # epoch > opt.epochs * opt.milestones[1]：如果当前轮数大于第二个里程碑轮数乘以总的训练轮数，表示已经达到了指定的训练阶段。
        # best_acc < acc：如果当前准确率大于之前的最佳准确率，表示有更好的准确率结果。
        # 如果这两个条件都满足，将执行以下操作：
        # weights_path：用于保存权重文件的路径，其中包含了当前轮数（epoch）、类型（type，这里是'best'）和准确率（acc）的信息。
        # print('saving weights file to {}'.format(weights_path))：打印保存权重文件的路径。
        # torch.save(net.state_dict(), weights_path)：保存模型的权重到指定的路径。
        # torch.save()：这是 PyTorch 库中的函数，用于将对象保存到磁盘上的文件中。在这里，我们将使用它来保存模型的权重。
        # net.state_dict()：这是一个模型对象（net）的方法，用于返回模型的状态字典（state dictionary）。
        # 模型的状态字典是一个 Python 字典对象，其中包含了模型的所有参数和缓冲区的当前数值。
        # weights_path：这是表示权重文件保存路径的字符串变量。它指定了要保存到的文件路径和文件名。
        # best_acc = acc：将当前准确率值更新为最佳准确率。
        # continue：继续进行下一轮的训练，跳过后续的代码。
        # 通过这段代码，可以定期保存在训练过程中准确率最好的模型权重文件，以便稍后进行评估、推理或恢复训练。只有当当前轮数超过特定里程碑轮数且准确率有所提高时，才会执行

        if epoch == opt.epochs:  # 最后一个也保存
            weights_path = checkpoint_path.format(epoch=epoch, train_path=opt.train_path, type='last', acc=acc,
                                                  jracc=jracc, erjracc=erjracc)
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    # end epoch
