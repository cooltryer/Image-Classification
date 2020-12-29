import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import argparse
import time
import sys
import numpy as np

from model.resnet import ResnetModel
# from model.model import resnet50
from utils.dataloader import LoadImage

transform = transforms.Compose([
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

target_names = [0, 1, 2, 3]

def train_one_epoch(epoch, model, train_dataloader, criterion, optimizer, device):
    # time1 = time.time()
    model.train()
    true = []
    pre = []
    losses = 0
    for image, label in tqdm(train_dataloader):
        # image, label = Variable(image, requires_grad=True).to(device), Variable(label.float(), requires_grad=True).to(device)
        image, label = Variable(image.float()).to(device), Variable(label).to(device)

        optimizer.zero_grad()  # 初始化梯度值
        output = model(image)
        

        loss = criterion(output, label.long())
        loss = loss
        loss.backward()  # 反向求解梯度
        losses += loss.item()
        optimizer.step()  # 更新参数

        pre_ = np.argmax(output.detach().cpu().numpy(), axis=1)
        true.extend(label.tolist())
        pre.extend(pre_.tolist())
        # acc = accuracy_score(true, pre)
        # print("Train Epoch({}): Loss:{} Acc:{}".format(epoch, loss, acc))
    # accuracy = accuracy_score(true, pre)
    classification = classification_report(true, pre, target_names=target_names)
    acc = accuracy_score(true, pre)
    print("Train Epoch({}): Loss:{} Acc:{} Res:\n{}".format(epoch, losses / len(train_dataloader), acc,  classification))
    # # time2 = time.time()
    # print(")

def test_one_epoch(epoch, model, test_dataloader, device):
    model.eval()
    true = []
    pre = []
    
    for image, label in tqdm(test_dataloader):
        # image, label = Variable(image, requires_grad=False, volatile=True).to(device), Variable(label.float(), requires_grad=False, volatile=True).to(device)
        image, label = Variable(image.float()).to(device), Variable(label).to(device)

        output = model(image)
        pre_ = np.argmax(output.detach().cpu().numpy(), axis=1)
        true.extend(label.tolist())
        pre.extend(pre_.tolist())

    classification = classification_report(true, pre)
    acc = accuracy_score(true, pre)
    print("Test Epoch({}): Acc:{} Res:\n{}".format(epoch, acc, classification))
    
def train(opt, device):
    print("Begin Trainning......")
    # 配置文件
    batch_size = opt.batch_size
    lr = opt.lr
    input_size = (opt.input_size, opt.input_size)
    epochs = opt.epochs
    train_path = opt.train_path
    test_path = opt.test_path
    weights = opt.weights
    model_path = opt.model_path

    # 数据读取
    train_datasets = LoadImage(transform, trainval=train_path)
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_datasets = LoadImage(transform, trainval=test_path)
    test_dataloader= DataLoader(test_datasets,  batch_size=batch_size//2, shuffle=False, num_workers=0, pin_memory=True)
    print("Train:", len(train_datasets), "      Test:", len(test_datasets))

    # 创建模型
    model = ResnetModel(num_classes=8, pretrained_path=model_path).to(device)
    # model = resnet50(pretrained=False).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    if True:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)   
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-6, weight_decay=5e-4)

    # 开始训练
    output_path = opt.output_path
    for epoch in range(epochs):
        train_one_epoch(epoch, model, train_dataloader, criterion, optimizer, device)
        test_one_epoch(epoch, model, test_dataloader, device)
        pth_path = output_path + "epoch" + str(epoch) + ".pth"
        torch.save(model.state_dict(), pth_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_path", type=str, default="train.txt")
    parser.add_argument("--test_path", type=str, default="val.txt")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="checkpoints/")
    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train(opt, device)
