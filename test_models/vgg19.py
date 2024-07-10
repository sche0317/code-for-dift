import torch
import torch.nn as nn
import torchvision.models as models

class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        model[36] = nn.Identity()
        self.net = model

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        pred = self.net(x)
        return pred