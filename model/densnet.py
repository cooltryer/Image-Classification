import torchvision.models.densenet as Densnet

import torch
import torch.nn as nn

import numpy as np

class ResnetModel(nn.Module):
    def __init__(self, num_classes, pretrained_path):
        super(ResnetModel, self).__init__()

        self.num_classes = num_classes
        self.pretrained_path = pretrained_path

        self.model = Densnet.densenet121(pretrained=True)
        if self.pretrained_path != "":
            pth = torch.load(pretrained_path)
            self.resnet50.load_state_dict(pth)
            print("Load pth success.....")
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("\t", name)
    
    def forward(self, x):
        x = self.model(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "/Users/hanxu/Documents/code/pytorch_model/resnet/resnet50-19c8e357.pth"
    model = ResnetModel(num_classes=5, pretrained_path=model_path).to(device)
    model.eval()  # 测试模式
    # print(model)
    x = torch.randn(1, 3, 224, 224)  # 模拟一张224*224的图片 batch_size=1 3通道
    pre = model(x)
    print(pre.shape, pre)

        
        