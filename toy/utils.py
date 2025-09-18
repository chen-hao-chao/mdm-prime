import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from PIL import Image
import os

def visualize_img_samples(samples, resolution, file):
    fig, axs = plt.subplots(1, 1, figsize=(20, 20))

    magma = mpl.colormaps['magma']
    newcmp = ListedColormap(magma(np.linspace(0.0, 0.75, 128)))

    H = axs.hist2d(samples[:, 0], samples[:, 1], 
                   range=[[0, resolution-1], [0, resolution-1]], 
                   bins=resolution)
    cmin = torch.quantile(torch.from_numpy(H[0]), 0.05).item()
    cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    _ = axs.hist2d(samples[:, 0], samples[:, 1], 
                   range=[[0, resolution-1], [0, resolution-1]], 
                   bins=resolution, norm=norm, cmap=newcmp)
    axs.set_aspect("equal")
    axs.axis("off")
    plt.tight_layout()
    plt.savefig(file)
    plt.savefig(file.split('.png')[0] + ".pdf")

def pngs_to_gif(folder_path, output_gif_path, duration=200, loop=0):
    # Get sorted list of PNG files
    png_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith('.png')
    ], key=lambda x: int(os.path.splitext(x)[0]))

    if not png_files:
        raise ValueError("No PNG files found in the folder.")

    # Load all images
    images = [Image.open(os.path.join(folder_path, f)) for f in png_files]

    # Save as GIF
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )

def encoder(x, base, bits, seq_length):
    with torch.no_grad():
        if bits == 1:
            return x
        
        x = x.reshape(x.shape[0], -1)
        mask = base ** torch.arange(bits - 1, -1, -1, device=x.device, dtype=x.dtype)
        
        digits = (x.unsqueeze(-1) // mask) % base
        
        x_code = digits.long()
        x_code = x_code.reshape(x.shape[0], bits * x.shape[1])

        return x_code
    
def decoder(b, base, bits, seq_length):
    with torch.no_grad():
        if bits == 1:
            return b
        
        b = b.reshape(b.shape[0], seq_length, bits)
        mask = base ** torch.arange(bits - 1, -1, -1, 
                                    device=b.device, dtype=b.dtype)
        output = (b * mask).sum(dim=-1)
        return output
