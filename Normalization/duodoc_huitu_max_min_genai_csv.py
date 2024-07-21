import os
import glob
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET  # 读取xml文件的包
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import csv

label_array = ["工况正常", "供液不足", "气体影响", "游动阀漏失", "固定阀漏失", "双阀漏失",
               "上碰泵", "下碰泵", "抽油杆断脱", "柱塞脱出工作筒", "稠油", "油井出砂", "错误功图"]

# 设置字体
matplotlib.rcParams['font.family'] = 'SimHei'

folder_path = 'D:\\aboutjiqixuexi\\indicator_diagram_data\\100j_2000p_xiugai_maxload_minload'  # 指定文件夹的路径

# 使用 glob 模块匹配文件夹中的所有 csv 文件
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))  # 这个列表是跟文件夹倒序的
num_batches = len(csv_files)  # 总共的组数

print(num_batches)
for w in range(99, 100):
    torch.cuda.empty_cache()  # 释放已经不在使用的缓存
    # print("第{}个文件".format(w))
    file_path = csv_files[w]
    csv_file_name = file_path.split('\\')[-1]  # 当前文件的名字
    print(file_path)
    print(csv_file_name)
    with open(file_path, 'r') as csvfile:
        # 创建 CSV 读取器
        reader = csv.reader(csvfile)
        print(type(reader))
        rows = list(reader)  # 可以把reader变为二维数组使用
        data_count = len(rows)  # 表格的行数,表格从1开始，但是rows结构从0开始

        # 获取相应字段的列号
        # 列号默认从0开始
        max_load = rows[0].index("max_load")
        min_load = rows[0].index("min_load")
        well_id = rows[0].index("well_id")
        moves = rows[0].index("moves")
        loads = rows[0].index("loads")
        gen_min = rows[0].index("gen_min")
        gen_max = rows[0].index("gen_max")

        print(moves)
        print(min_load)

        # print(3487)
        # 遍历每一行
        for i in range(1, data_count):  # 这次遍历会直接从表头下面那一行开始,到最后一行
            print("第{}个文件".format(w))
            print("第{}个图".format(i))
            # print(file_path)
            # print(98)
            # 获取相应字段的数值
            x_data = rows[i][moves]
            # 获取相应字段的数值
            y_data = rows[i][loads]
            # print(type(x_data))
            # print(x_data)
            # 分割并转换x_data和y_data
            x_values = [float(x) for x in x_data.split(';') if x]
            y_values = [float(y) for y in y_data.split(';') if y]
            # 假设示功图数据是一个形状为(200, 2)的张量，其中每一行表示一个示功图数据点的横坐标和纵坐标
            data = torch.randn(200, 2)
            # print(y_values)
            # 将x_values和y_values组合成data对象
            for x in range(len(x_values)):
                data[x][0] = x_values[x]
                data[x][1] = y_values[x]

            x_scaler = MinMaxScaler()
            x_normalized = x_scaler.fit_transform(data[:, 0].reshape(-1, 1))

            # 对 y 维度进行自定义的最大最小值归一化，并映射到 0 到 1 的范围
            if float(rows[i][gen_min]) > float(rows[i][min_load]):
                y_min = float(rows[i][min_load])  # 自定义的最小值
            else:
                y_min = float(rows[i][gen_min])

            if float(rows[i][gen_max]) > float(rows[i][max_load]):
                y_max = float(rows[i][gen_max])  # 自定义的最大值
            else:
                y_max = float(rows[i][max_load])

            y_scaled = (data[:, 1] - y_min) / (y_max - y_min)
            y_normalized = y_scaled.reshape(-1, 1)

            # 将 x 和 y 维度归一化后的结果合并
            normalized_data = np.concatenate((x_normalized, y_normalized), axis=1)

            # 创建一个新的图形
            plt.figure()

            # 提取x坐标和y坐标
            x = normalized_data[:, 0]
            y = normalized_data[:, 1]

            # 绘制连线图
            plt.plot(x, y)
            plt.plot([x[0], x[-1]], [y[0], y[-1]], color='#1f77b4ff')  # 连接第一个点和最后一个点，形成封闭图形
            # [x[0], x[-1]]表示一个包含第一个点和最后一个点的横坐标列表。x[0]是第一个点的横坐标，x[-1]是最后一个点的横坐标。
            # label = int(row[label_0])
            # print(label)
            # 添加标题和标签

            plt.xlabel("X")
            plt.ylabel("Y")
            plt.axis('off')  # 关闭坐标轴和其他信息
            # 保存图像到指定路径

            #
            # save_path = ("D:/aboutjiqixuexi/shigongtuyuchuli/dataset/{}/{}{}{}{}.png"
            #              .format(erjidoc, "l_", label, "_", step))
            # erjidoc = "{}{}".format(label, label_array[label - 15])  # 二级文件目录

            # 基础目录
            file_path_0 = "D:\\aboutjiqixuexi\\indicator_diagram_data\\img_100j_2000p_xiugai_maxload_minload"
            erjidoc = "img_{}".format(csv_file_name)  # 二级文件目录
            filename = "{}_{}.png".format(i+1, rows[i][well_id])  # 指定文件名

            subdirectory_path = os.path.join(file_path_0, erjidoc)
            # 路径不存在就创建
            if not os.path.exists(subdirectory_path):
                os.makedirs(subdirectory_path)

            # 构建文件的完整路径
            save_path = os.path.join(subdirectory_path, filename)
            print(save_path)
            #  显示自定义最大最小归一化的图像
            plt.ylim(0, 1)  # 设置 Y 轴范围为 0 到 1

            plt.savefig(save_path)  # 保存
            # 打开彩色图片
            image = Image.open(save_path)
            # print(image.size)
            # 将彩色图像转换为灰度图像
            gray_image = image.convert('L')
            # 保存灰度图像
            gray_image.save(save_path)
            # 打印归一化后的数据
            # print(normalized_data)
            # plt.show()
