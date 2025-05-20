import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet

class UNet3DExpert(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=1,
        )

    def forward(self, x):
        return self.model(x)