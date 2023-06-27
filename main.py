import argparse
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

# 定义超参数
parser = argparse.ArgumentParser(description='Knowledge Distillation with ResNet')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
parser.add_argument('--temperature', type=float, default=5.0, help='Temperature parameter for knowledge distillation')
parser.add_argument('--alpha', type=float, default=0.5, help='Weight for combining soft and hard targets')
parser.add_argument('--data-dir', type=str, default='/data1/zhengshuaijie/data', help='Path to the directory containing the dataset')
parser.add_argument('--output-dir', type=str, default='/home/zhengshuaijie/repository/knowledge_Distillation/output', help='Path to the directory for saving the model')
parser.add_argument('--device', type=str, default='cuda:3', help='Device to use for training (e.g. cpu, cuda)')
args = parser.parse_args()

def collate_fn(batch):
    inputs, targets = tuple(zip(*batch))
    inputs = torch.stack(inputs,dim=0)
    print(inputs.shape)

    targets = torch.stack(targets,dim=0)
    print(targets.shape)

    return inputs, targets

# 数据变换和加载器
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = datasets.CocoDetection(root=os.path.join(args.data_dir,'train2017'), annFile=os.path.join(args.data_dir, 'annotations/instances_train2017.json'), transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

test_dataset = datasets.CocoDetection(root=os.path.join(args.data_dir,'val2017'), annFile=os.path.join(args.data_dir, 'annotations/instances_val2017.json'), transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

# 教师模型和学生模型
teacher_model = resnet152(pretrained=True)
teacher_model.fc = nn.Linear(2048, len(train_dataset.coco.cats))
teacher_model.to(args.device)
teacher_model.eval()

student_model = resnet18(pretrained=True)
student_model.fc = nn.Linear(512, len(train_dataset.coco.cats))
student_model.to(args.device)
criterion = nn.KLDivLoss().to(args.device)
optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# 辅助函数
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample([x.size(0), 1, 1, 1]).to(x.device)
        index = torch.randperm(x.size(0)).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    else:
        return x, y, None, None

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    if lam is not None:
        return (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()
    else:
        return criterion(pred, y_a)

def label_smooth(target, num_classes, smoothing=0.1):
    confidence = 1 - smoothing
    label_shape = torch.Size((target.size(0), num_classes))
    smoothed_labels = torch.full(size=label_shape, fill_value=smoothing / (num_classes - 1)).to(target.device)
    smoothed_labels.scatter_(1, target.unsqueeze(1), confidence)
    return smoothed_labels

# 训练和测试函数
def train(epoch):
    student_model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha)
        optimizer.zero_grad()
        outputs = student_model(mixed_inputs)
        if targets_b is None:
            loss = criterion(outputs, label_smooth(targets_a, train_dataset.coco.cats))
        else:
            soft_targets = torch.softmax(teacher_model(inputs) / args.temperature, dim=1)
            soft_targets = soft_targets.detach()
            soft_targets = mixup_data(soft_targets, soft_targets[index, :], lam)[0]
            soft_targets = label_smooth(soft_targets, train_dataset.coco.cats)
            loss = mixup_criterion(criterion, outputs, label_smooth(targets_a, train_dataset.coco.cats), soft_targets, lam)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_description('Epoch: [{}/{}], Loss: {:.4f}, Acc: {:.2f}%'.format(epoch, args.epochs, train_loss / (batch_idx + 1), 100. * correct / total))

def test(epoch):
    student_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = student_model(inputs)
            loss = criterion(outputs, label_smooth(targets, train_dataset.coco.cats))
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_description('Epoch: [{}/{}], Loss: {:.4f}, Acc: {:.2f}%'.format(epoch, args.epochs, test_loss / (batch_idx + 1), 100. * correct / total))
    return 100. * correct / total

# 训练主函数
def main():
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(epoch)
        acc = test(epoch)
        end_time = time.time()
        print('Time taken for epoch {} is {:.2f} seconds'.format(epoch, end_time - start_time))
        if acc > best_acc:
            best_acc = acc
            torch.save(student_model.state_dict(), os.path.join(args.output_dir, 'resnet18_best.pth'))
    print('Best test accuracy: {:.2f}%'.format(best_acc))

if __name__ == '__main__':
    main()