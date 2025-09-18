# source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import os
import numpy as np
from torch.utils.data import Dataset
import pickle
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def process(d, img_size=32):
    x = d['data']
    y = d['labels']
    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    y = np.asarray(y)
    img_size2 = img_size * img_size
    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    
    return x, y

class ImageNet32Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform

        print("loading ImageNet...")
        images = None
        labels = None
        if split == "train":
            for idx in range(10):
                data_file = os.path.join(root_dir, "train", 'train_data_batch_')
                d = unpickle(data_file + str(idx+1))
                image, label = process(d)
                labels = np.concatenate((labels, label), axis=0) if labels is not None else label
                images = np.concatenate((images, image), axis=0) if images is not None else image
        elif split == "val":
            data_file = os.path.join(root_dir, 'val', 'val_data')
            d = unpickle(data_file)
            images, labels = process(d)
        else:
            raise ValueError("Set not recognized.")
        
        print("finish loading ImageNet...")
        
        self.data = list(zip(images, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, labels = self.data[idx]
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)

        return image, labels