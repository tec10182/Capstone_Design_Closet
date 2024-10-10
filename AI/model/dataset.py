import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os


class CustomDataset(Dataset):
    def __init__(self, image_dir, meta_dir, transform=None):
        self.image_dir = image_dir
        self.meta_dir = meta_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")

        meta = pd.read_csv(self.meta_dir)
        x = meta.loc[meta["img_name"] == self.image_filenames[idx], "x"].values[0]
        y = meta.loc[meta["img_name"] == self.image_filenames[idx], "y"].values[0]
        w = meta.loc[meta["img_name"] == self.image_filenames[idx], "width"].values[0]
        h = meta.loc[meta["img_name"] == self.image_filenames[idx], "height"].values[0]

        x = x.split(",")
        y = y.split(",")
        w = w.split(",")
        h = h.split(",")

        cropped_images = []

        for i in range(len(x)):
            left = int(x[i])
            upper = int(y[i])
            right = left + int(w[i])
            lower = upper + int(h[i])

            cropped_image = image.crop((left, upper, right, lower))

            if self.transform:
                cropped_image = self.transform(cropped_image)

            cropped_images.append(cropped_image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.stack(cropped_images).to(device), len(
            cropped_images
        )  # (N, C, H, W)
