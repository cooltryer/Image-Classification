# 找到有问题的图片
import cv2
import os
from skimage import io
from tqdm import tqdm

def image_label(path="./data/val_data/"):
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

if __name__ == "__main__":
    image_list, label = image_label()
    err = []
    for i in tqdm(image_list):
        try:
            io.imread(i)
        except Exception as e:
            err.append(i)
    print(err)