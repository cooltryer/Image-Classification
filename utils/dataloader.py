import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas as pd
import cv2
from PIL import Image
import os

def image_label_kaggle(path="./utils/train.csv"):
    # 得到每一个image的路径及其对应的标签，以列表形式返回
    data_csv = pd.read_csv(path)
    image_list = data_csv["image_id"].values.tolist()
    label_list = data_csv["label"].values.tolist()
    image_path = []
    base_path = "/".join(path.split("/")[: -1]) + "/train_images/"
    # print(base_path)
    for e in image_list:
        image_path.append(base_path + e)

    return image_path, label_list

def image_label(path="./data/train_data/"):
    # 得到每一个image的路径及其对应的标签，以列表形式返回
    image_path = []
    label_list = []
    
    label_path = os.listdir(path)
    for label in label_path:
        img_path = path + label + "/"
        img_all_path = os.listdir(img_path)
        for image in img_all_path:
            image_path.append(img_path + "/" + str(image))
            label_list.append(int(label))

    return image_path, label_list

def read_txt(path="train.txt"):
    data = open(path)
    lines = data.readlines()
    
    image_path = []
    label_list = []
    for line in lines:
        words = line.split(".jpg ")
        image = words[0] + ".jpg"
        label = int(words[-1])
        
        image_path.append(image)
        label_list.append(label)
    
    return image_path, label_list

class LoadImage(Dataset):
    def __init__(self, transfrom, trainval="train.txt"):
        super(LoadImage, self).__init__()

        self.trainval = trainval
        # self.image_path, self.label_list = read_txt(path=self.trainval)
        self.image_path, self.label_list = image_label_kaggle(path=trainval)
        self.transfrom = transfrom
        # self.input_size = input_size

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, item):
        image = self.image_path[item]
        label = self.label_list[item]
        # print("测试", image)
        # image = cv2.imread(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 用opencv读取需要转成RGB图像
        # image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_CUBIC)
        image = Image.open(image).convert('RGB')
        # print(image.size)
        if self.transfrom:
            image = self.transfrom(image)

        return image, label

if __name__ == "__main__":
    print("Test LoadImage......")
    # read_txt()
    transform = transforms.Compose([
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    input_size = (224, 224)
    batch_size = 1
    train_datasets = LoadImage(transform, trainval="./data/5train.csv")
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=False, num_workers=0)
    for x,y in train_dataloader:
        print(x.shape, y)

    
    
