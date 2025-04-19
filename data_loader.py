from random import random

from torchvision import transforms
from torch.utils.data import Dataset
import os
from torch import tensor, float
from PIL import Image


class SignaturePairDataset(Dataset):
    def __init__(self, genuine_dir, forged_dir, transform=None):
        self.genuine_images = sorted(
            [os.path.join(genuine_dir, file) \
             for file in os.listdir(genuine_dir) if file.lower().endswith(".png")]
        )
        self.forged_images = sorted(
            [os.path.join(forged_dir, file) \
             for file in os.listdir(forged_dir) if file.lower().endswith(".png")]
        )
        # self.transform = transform if \
        #     transform is not None else transforms.ToTensor()

        # 0 Transform image sizes
        # Keeps each sample at 16k pixels, so an 8–16GB GPU
        # can process batch sizes of 16–64 without OOM.
        # Small enough for fast data loading and augmentation on a CPU.
        # Preserves the finer curves in a handwritten signature.
        fixed_size = (128, 128)

        self.transform = transform if \
            transform is not None else \
            transforms.Compose([
                transforms.Resize(fixed_size),
                transforms.ToTensor()
            ])
        self.pairs = []
        self.labels = []

        # Generate positive pairs
        for i in range(len(self.genuine_images)):
            for j in range(i + 1, len(self.genuine_images)):
                self.pairs.append((self.genuine_images[i], self.genuine_images[j]))
                self.labels.append(1.0)

        # Generate negative pairs
        for genuine_image in self.genuine_images:
            for forged_image in self.forged_images:
                self.pairs.append((genuine_image, forged_image))
                self.labels.append(0.0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path1, path2 = self.pairs[index]

        # Load the images from disk
        img1 = Image.open(path1).convert('L')  # use 'L' for grayscale; 'RGB' for color
        img2 = Image.open(path2).convert('L')

        # Apply the transform to convert images to tensors
        image1_transformed = self.transform(img1)
        image2_transformed = self.transform(img2)

        label = tensor(self.labels[index], dtype=float)
        return image1_transformed, image2_transformed, label
