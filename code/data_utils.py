# Imports
import cv2
import pandas as pd

# PyTorch Imports
from torch.utils.data import Dataset
from torchvision import transforms



# Class: FaceDataset
class FaceDataset(Dataset):
    def __init__(self, file_name, is_train, input_size=224, pre_mean=[0.5, 0.5, 0.5], pre_std=[0.5, 0.5, 0.5]):
        self.data = pd.read_csv(file_name)
        self.is_train = is_train
        
        self.train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize([input_size, input_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=pre_mean, std=pre_std),
             ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize([input_size, input_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=pre_mean, std=pre_std),
             ]
        )


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        image_path = self.data.iloc[index, 0]
        label_str = self.data.iloc[index, 1]
        label = 1 if label_str == 'bonafide' else 0

        image=cv2.imread("../Morphing/"+image_path)
        try:
            if self.is_train:
                image = self.train_transform(image)
            else:
                image = self.test_transform(image)
        except ValueError:
            print(image_path)

        return image, label
