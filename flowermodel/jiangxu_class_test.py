class MyClass:
    def __init__(self, my_float, my_string):
        self.my_float = my_float
        self.my_string = my_string

# 创建一个包含类对象的数组
my_objects = [
    MyClass(3.14, "Object 1"),
    MyClass(1.23, "Object 2"),
    MyClass(2.56, "Object 3"),
    MyClass(0.5, "Object 4"),
    MyClass(2.0, "Object 5"),
    MyClass(1.0, "Object 6"),
    # 添加更多对象...
]

# 指定排序范围，对第2位到第5位进行排序
start_index = 1  # 范围起始位置的索引（从0开始）
end_index = 5    # 范围结束位置的索引（不包含）

# 选择要排序的子数组
sub_array = my_objects[start_index:end_index]

# 对子数组进行排序
sub_array.sort(key=lambda obj: obj.my_float, reverse=True)

# 将排序后的子数组放回原数组的对应位置
my_objects[start_index:end_index] = sub_array

# 打印排序后的结果
for obj in my_objects:
    print(obj.my_float, obj.my_string)