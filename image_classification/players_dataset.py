from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision.transforms import Compose, ToTensor, Lambda


def custom_resize(image):
    return ImageOps.pad(image, (224, 224), color=(0, 0, 0))

class PlayersDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.labels = []
        self.root = root
        self.image_paths = []
        self.transform = transform
        self.categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

        if train:
            data_path = os.path.join(root, 'train')
        else:
            data_path = os.path.join(root, 'val')

        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for file in os.listdir(data_files):
                image_path = os.path.join(data_files, file)
                self.image_paths.append(image_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.image_paths[index]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label


if __name__ == '__main__':
    root = 'player_images'

    train_transform = Compose([
        Lambda(custom_resize),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = PlayersDataset(root=root, train=True, transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    image, label = train_dataset[200]
    image = image.permute(1, 2, 0).numpy()

    plt.title(f"Label: {label}")
    plt.imshow(image)
    plt.axis('off')
    plt.show()

