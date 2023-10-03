from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import affine, adjust_contrast, adjust_brightness, resize, adjust_saturation
import torch
import os
import pandas as pd
from numpy import genfromtxt
import numpy as np


def point_rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = torch.tensor([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = torch.atleast_2d(origin)
    p = torch.atleast_2d(p)
    return torch.squeeze((R @ (p.T-o.T) + o.T).T)


class KeypointDataset(Dataset):
    def __init__(self, filelist, root_dir, transforms=False):
        files_path = os.path.join(root_dir, filelist)
        self.files = pd.read_csv(files_path)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = str(self.files.iloc[idx, 0])
        file_path = os.path.join(self.root_dir + 'img/', fname + '.png')
        ann_path = os.path.join(self.root_dir + 'ann/', fname + '.txt')
        img = read_image(file_path, ImageReadMode.RGB).to(torch.float32) / 255
        label_array = genfromtxt(ann_path, delimiter=' ')
        label = torch.from_numpy(label_array).to(torch.float32)
        label = label.reshape(7, 2)

        h, w = img.shape[1:]
        new_w, new_h = 80, 80
        img = resize(img, [new_h, new_w])
        label *= torch.tensor([new_w / w, new_h / h])

        if self.transforms:
            img = adjust_contrast(img, np.random.uniform(0.4, 1.6))
            img = adjust_brightness(img, np.random.uniform(0.4, 1.6))
            img = adjust_saturation(img, np.random.uniform(0.4, 1.6))

            theta = np.random.randint(-15, 15)
            theta_rad = np.deg2rad(-theta)

            translate = np.random.uniform(-0.1, 0.1, 2)
            wt = new_w * translate[0]
            ht = new_h * translate[1]
            img = affine(img, angle=theta, translate=[int(wt), int(ht)], scale=1.0, shear=[0.0, 0.0])

            R = torch.tensor([[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]]).to(torch.float32)
            center = torch.tensor([new_w / 2, new_h / 2]).to(torch.float32)
            label = (label - center) @ R + center

            label[:, 0] += wt
            label[:, 1] += ht

        label = label.flatten()

        return img, label