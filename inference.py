import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.resnet import ResnetModel
from utils.dataloader import LoadImage
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np

from model.resnet import ResnetModel

transform = transforms.Compose([
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

def inference(device, path="val.txt"):
    inference_datasets = LoadImage(transform, trainval=path)
    inference_dataloader = DataLoader(inference_datasets, batch_size=8, shuffle=False, num_workers=0)

    model = ResnetModel(num_classes=8, pretrained_path="").to(device)
    model.load_state_dict(torch.load("epoch0.pth"))
    model.eval()

    pre = []
    true = [] 
    for image, label in tqdm(inference_dataloader):
        image, label = image.to(device),label.float().to(device)
        # print(image)

        output = model(image).float()
        pre_ = np.argmax(output.detach().cpu().numpy(), axis=1)
        true.extend(label.tolist())
        pre.extend(pre_.tolist())
    print(classification_report(true, pre))
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inference(device)