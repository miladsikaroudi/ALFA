import pandas as pd
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from utils.augmentation_utils import *



ImageFile.LOAD_TRUNCATED_IMAGES = True




class StandardDataloader(Dataset):
    def __init__(self, src_path, sample_paths, class_labels, domain_label=-1):
        self.image_transformer = T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.3, 0.3),
                T.RandomGrayscale(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.self_supervision_transformer = T.Compose([
                    HEDJitter(theta=0.05),
                    RandomAffine(degrees=[-10,10], translate=[0, 0.1], shear=[-1, 1, -1, 1], fillcolor=(0, 0, 0)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        self.src_path = src_path
        self.domain_label = domain_label
        self.sample_paths, self.class_labels = sample_paths, class_labels

    def get_image(self, sample_path):
        img = Image.open(sample_path).convert("RGB")
        return self.image_transformer(img), self.self_supervision_transformer(img)

    def __len__(self):
        return len(self.sample_paths)


    def __getitem__(self, index):
        sample, sample_ssl = self.get_image(self.src_path + self.sample_paths[index])
        class_label = self.class_labels[index]
        sample_path = self.sample_paths[index]


        return sample, sample_ssl, class_label, self.domain_label, sample_path


class StandardValDataloader(StandardDataloader):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self.image_transformer = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
