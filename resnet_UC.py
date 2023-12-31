import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # new import
from tqdm import tqdm

# 定义命令行参数
parser = argparse.ArgumentParser(description='Train a ResNet34 model for UC Merced Land-Use dataset')
parser.add_argument('--data_dir', type=str, default='/data1/zhengshuaijie/UCMerced_LandUse', help='path to the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_classes', type=int, default=21, help='number of classes')
args = parser.parse_args()

# 定义训练和验证的转换器
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir,'train'), transform=train_transforms)
val_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir,'val'), transform=val_transforms)
# 定义批量大小和数据加载器
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# # 定义模型
# class ResNet34(nn.Module):
#     def __init__(self, num_classes=21):
#         super(ResNet34, self).__init__()
#         self.resnet = models.resnet34(pretrained=True)
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_ftrs, num_classes)

#     def forward(self, x):
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)

#         x = self.resnet.layer1(x)
#         x = self.resnet.layer2(x)
#         x = self.resnet.layer3(x)
#         x = self.resnet.layer4(x)

#         x = self.resnet.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.resnet.fc(x)

#         return x

# 初始化模型和损失函数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = models.resnet34(num_classes=args.num_classes).to(device)
model = models.resnet34()
model.fc= nn.Linear(model.fc.in_features,args.num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 初始化 Tensorboard
writer = SummaryWriter()

print(model)#调用模型然后计算模型体系

# 训练模型
best_val_acc = 0.0  # 记录最好的验证准确率
for epoch in range(args.num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0

    # 使用 tqdm 显示训练进度条
    with tqdm(train_loader, unit='batch') as t:
        t.set_description(f'Epoch [{epoch+1}/{args.num_epochs}]')
        model.train()  # 训练模式
        for i, (images, labels) in enumerate(t):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)

            # 更新进度条
            t.set_postfix(loss=loss.item(), acc=train_acc.item()/len(train_loader.dataset))


        writer.add_scalar('train/loss', loss.item(), epoch)
        writer.add_scalar('train/accuracy', train_acc.item()/len(train_loader.dataset), epoch)

        # 保存模型权重
        # torch.save(model.state_dict(), f"model_{epoch+1}.pth")

    model.eval()  # 评估模式
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_acc / len(val_loader.dataset)

    # 输出每个 epoch 的结果
    print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 记录验证损失和准确率到 Tensorboard
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/accuracy', val_acc, epoch)

    # 选择最好的权重
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = model.state_dict()

    # 每十轮保存一次，保存最好的权重
    if int(epoch+1) % 10 == 0:
        torch.save(best_weights, "best_model.pth") 
        print('保存模型参数')

    

# 关闭 Tensorboard
writer.close()