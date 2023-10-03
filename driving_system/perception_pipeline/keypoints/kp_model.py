import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class ConvLayer(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size, stride, padding),
            nn.BatchNorm2d(n_out, eps=1e-05, momentum=0.05, affine=True),
            Mish(),
        )

    def forward(self, x):
        return self.seq(x)


class DoubleConvLayer(nn.Module):
    def __init__(self, n_in, n_out1, n_out2, kern1=3, kern2=3, stride1=1, stride2=1, pad1=1, pad2=1):
        super(DoubleConvLayer, self).__init__()
        self.seq = nn.Sequential(
            ConvLayer(n_in, n_out1, kern1, stride1, pad1),
            ConvLayer(n_out1, n_out2, kern2, stride2, pad2)
        )

    def forward(self, x):
        return self.seq(x)


class ResBlock2(nn.Module):
    def __init__(self, n_in, n_out):
        super(ResBlock2, self).__init__()
        self.first = ConvLayer(n_in, n_out)
        self.branch = DoubleConvLayer(n_out, n_out, n_out)

    def forward(self, x):
        x = self.first(x)
        return x + self.branch(x)


class KeypointModel(nn.Module):
    def __init__(self):
        super(KeypointModel, self).__init__()
        self.seq = nn.Sequential(
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            DoubleConvLayer(3, 64, 128, stride1=1, stride2=2),
            ResBlock2(128, 128),
            DoubleConvLayer(128, 256, 512, stride1=1, stride2=2),
            ResBlock2(512, 512),
        )
        self.drop = nn.Dropout(0.10)
        self.fc = nn.Linear(512, 14)

    def forward(self, x):
        x = self.seq(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)