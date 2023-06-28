import argparse
from sklearn.metrics import confusion_matrix
import torch
from torchvision.models import resnet34 
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from  torchvision.models import resnet18, resnet152

# 定义命令行参数
parser = argparse.ArgumentParser(description='Train a ResNet34 model for UC Merced Land-Use dataset')
parser.add_argument('--data_dir', type=str, default='/data1/zhengshuaijie/UCMerced_LandUse', help='path to the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_classes', type=int, default=21, help='number of classes')
args = parser.parse_args()

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = resnet34(num_classes=args.num_classes).to(device)
model.load_state_dict(torch.load('best_model.pth'))

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 加载测试集
test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir,'test'), transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# 预测测试集并计算混淆矩阵
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)