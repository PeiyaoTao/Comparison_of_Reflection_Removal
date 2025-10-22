# Ziyan Chen, CS7180, 19th Oct 2025
# This code is based on https://github.com/Vandermode/ERRNet
# Add your custom network here
from .default import DRNet
import torch.nn as nn


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, use_fpn=True, use_eca=True, **kwargs)
