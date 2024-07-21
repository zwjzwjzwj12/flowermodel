# 定义变换后的 sigmoid 函数
import numpy as np


def transformed_sigmoid(x):
    return 1 / (1 + np.exp(-x * 15)) - 0.5


# 定义对下碰泵的膨胀分布函数
def pengzhangfenbu(x, center):
    qujian2 = [0, 0.5]
    q2len = qujian2[1] - qujian2[0]
    daqujian2 = [center, 1]
    dq2len = daqujian2[1] - daqujian2[0]

    if x > center:
        y = transformed_sigmoid((x - center) / (dq2len / q2len)) * (dq2len / q2len) + center
        return y
    else:
        return x


print(pengzhangfenbu(0.4, 0.5))
