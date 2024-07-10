import os
import sys
import torch
import torch.nn as nn
import numpy as np
from .vgg19 import vgg
import torch.nn.functional as functional

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.register_buffer('shift2', torch.Tensor([1.0, 1.0, 1.0])[None, :, None, None])
        self.register_buffer('scale2', torch.Tensor([2.0, 2.0, 2.0])[None, :, None, None])

    def forward(self, x):
        x = (x + self.shift2) / self.scale2
        x = (x - self.shift) / self.scale
        return x

def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

def spatial_average(x, keepdim=True):
    return x.mean([1, 2, 3], keepdim=keepdim)

class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        # self.scaling_layer = ScalingLayer()
        self.net = vgg()
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, x, y=None):
        x = self.net(x)
        y = self.net(y)
        res = []
        for i1, i2 in zip(x, y):
            i1 = i1.unsqueeze(0)
            i2 = i2.unsqueeze(0)
            f1, f2 = normalize_tensor(i1), normalize_tensor(i2)
            diff = (f1 - f2) ** 2
            avg = spatial_average(diff, keepdim=False)
            res.append(avg)
        res = torch.cat(res, dim=0).sum()
        return res
