import os
import shutil
import random
from sklearn.model_selection import train_test_split

origin_data_dir = '/data1/zhengshuaijie/UCMerced_LandUse/Images'
target_data_dir = '/data1/zhengshuaijie/UCMerced_LandUse/'
train_dir = os.path.join(target_data_dir, 'train')
val_dir = os.path.join(target_data_dir, 'val')
test_dir = os.path.join(target_data_dir, 'test')

# 获取所有图像的文件名和标签
image_filenames = []
labels = []
for label, class_name in enumerate(sorted(os.listdir(origin_data_dir))):
    class_dir = os.path.join(origin_data_dir, class_name)
    for filename in sorted(os.listdir(class_dir)):
        if filename.endswith('.tif'):
            image_filenames.append(os.path.join(class_dir, filename))
            labels.append(label)

# 划分数据集为训练集、验证集和测试集
train_val_filenames, test_filenames, train_val_labels, test_labels = train_test_split(image_filenames, labels, test_size=0.2, random_state=42, stratify=labels)
train_filenames, val_filenames, train_labels, val_labels = train_test_split(train_val_filenames, train_val_labels, test_size=0.25, random_state=42, stratify=train_val_labels)

# 创建训练集、验证集和测试集目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 将图像文件复制到相应的目录中
for filenames, labels, directory in [(train_filenames, train_labels, train_dir), (val_filenames, val_labels, val_dir), (test_filenames, test_labels, test_dir)]:
    for filename, label in zip(filenames, labels):
        class_name = os.path.basename(os.path.dirname(filename))
        dst_dir = os.path.join(directory, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(filename, os.path.join(dst_dir, os.path.basename(filename)))

print("数据集划分完成！")