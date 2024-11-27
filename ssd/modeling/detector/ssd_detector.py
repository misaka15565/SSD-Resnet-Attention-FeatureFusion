from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head

import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        
        # 1x1 卷积用于横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])
        
        # 3x3 卷积用于输出特征图
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])
    
    def forward(self, features):
        num_features = len(features)
        fpn_features = [None] * num_features

        # 最顶层特征直接处理
        fpn_features[-1] = self.output_convs[-1](self.lateral_convs[-1](features[-1]))
    
        # 自顶向下逐层融合
        for i in range(num_features - 2, -1, -1):
            # 横向连接
            lateral = self.lateral_convs[i](features[i])
        
            # 上采样并调整形状
            upsampled = F.interpolate(fpn_features[i + 1], size=lateral.shape[2:], mode='nearest')
        
            # 确保形状一致后相加
            fpn_features[i] = self.output_convs[i](lateral + upsampled)

        return fpn_features


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.fpn = FPN(in_channels=[512,1024,2048,1024,512,256],out_channels=256)
        self.box_head = build_box_head(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        f2 = self.fpn(features)
        detections, detector_losses = self.box_head(f2, targets)
        if self.training:
            return detector_losses
        return detections
