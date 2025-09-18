import torch
import numpy as np
from PIL import Image


def convert_img_to_array(path, size):
    img = Image.open(path).convert("L").resize((size, size)).rotate(270, Image.NEAREST)
    numpy_img = np.asarray(img)
    # numpy_img = -numpy_img + 255
    return numpy_img


def img_sampler(p, size: int = 128, batch_size: int = 200, device: str = "cpu"):
    all_index = np.arange(size * size).reshape(size, size)
    samples = np.random.choice(all_index.flatten(), batch_size, p=p.flatten())
    index = np.unravel_index(samples, all_index.shape)
    index = np.column_stack(index)
    return torch.tensor(index, device=device).long()
