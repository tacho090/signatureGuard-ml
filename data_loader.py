from random import random

from torch.utils.data import Dataset


class SignaturePairDataset(Dataset):
    def __init__(self, genuine_images, forged_images):
        self.genuine_images = genuine_images
        self.forged_images = forged_images
        self.pairs = []
        self.labels = []

        # Generate positive pairs
        for i in range(len(genuine_images)):
            for j in range(i + 1, len(genuine_images)):
                self.pairs.append((genuine_images[i], genuine_images[j]))
                self.labels.append(1.0)

        # Generate negative pairs
        for genuine in genuine_images:
            for forged in forged_images:
                self.pairs.append((genuine, forged))
                self.labels.append(0.0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img, img2 = self.pairs[index]
        label = self.labels[index]
        return img, img2, label
