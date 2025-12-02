from .meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN, build_model
from .roi_heads import (
    ROI_BOX_HEAD_REGISTRY, ROI_HEADS_REGISTRY, ROIHeads, StandardROIHeads, build_box_head,
    build_roi_heads)
"""
class AdvancedCBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernels=[3,7,11]):
        super().__init__()
        self.channels = channels
        # --- 通道注意力 ---
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel_residual = nn.Parameter(torch.ones(1, channels, 1, 1))  # learnable residual

        # --- 空间注意力 ---
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(2, 1, k, padding=k//2, bias=False) for k in spatial_kernels
        ])
        # 可学习权重融合多尺度
        self.spatial_weights = nn.Parameter(torch.ones(len(spatial_kernels)) / len(spatial_kernels))

    def forward(self, x):
        N, C, H, W = x.size()

        # --- 通道注意力 ---
        avg = self.avg_pool(x).permute(0,2,1,3).contiguous().view(-1,1,C)
        max = self.max_pool(x).permute(0,2,1,3).contiguous().view(-1,1,C)
        channel_attn = self.sigmoid(self.conv1d(avg) + self.conv1d(max))
        channel_attn = channel_attn.view(N, C, 1, 1)
        x_ca = x * channel_attn + self.channel_residual * x  # 通道注意力残差

        # --- 空间注意力 ---
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool,_ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # [N,2,H,W]
        spatial_maps = torch.stack([conv(spatial_input) for conv in self.spatial_convs], dim=0)  # [K,N,1,H,W]
        spatial_attn = (spatial_maps * self.spatial_weights.view(-1,1,1,1,1)).sum(dim=0)
        spatial_attn = self.sigmoid(spatial_attn)

        # --- 输出融合 ---
        out = x_ca * spatial_attn + x  # 残差保留原始信息
        return out
        import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassGuidedDecoupledAttention(nn.Module):
    def __init__(self, channels, num_classes, reduction=16, spatial_kernels=[3, 7]):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        
        # === 类别原型嵌入 ===
        self.class_embed = nn.Embedding(num_classes, channels)
        
        # --- 通道注意力（类别引导） ---
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # --- 空间注意力（多尺度 + 类别 gating） ---
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(2, 1, k, padding=k // 2, bias=False) 
            for k in spatial_kernels
        ])
        self.spatial_weights = nn.Parameter(
            torch.ones(len(spatial_kernels)) / len(spatial_kernels)
        )
        self.spatial_gate = nn.Linear(channels, 1)  # 类别 gating

    def forward(self, x, cls_ids=None):
        
        x: [N, C, H, W]
        cls_ids: [N] 每个样本的类别id（训练时可用，推理时可用预测类别）
        
        N, C, H, W = x.size()
        
        # ===== 类别嵌入 =====
        if cls_ids is not None:
            class_vec = self.class_embed(cls_ids)  # [N, C]
        else:
            # 如果推理阶段没有类别信息，就用均值
            class_vec = self.class_embed.weight.mean(
                dim=0, keepdim=True
            ).repeat(N, 1)
        
        # --- 通道注意力 ---
        avg = self.avg_pool(x).view(N, C)
        max = self.max_pool(x).view(N, C)
        channel_attn = avg + max + class_vec
        channel_attn = self.fc2(F.relu(self.fc1(channel_attn))).view(N, C, 1, 1)
        channel_attn = self.sigmoid(channel_attn)
        x_ca = x * channel_attn + x  # 残差增强
        
        # --- 空间注意力 ---
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # [N,2,H,W]
        
        spatial_maps = torch.stack(
            [conv(spatial_input) for conv in self.spatial_convs], 
            dim=0
        )  # [K,N,1,H,W]
        spatial_attn = (spatial_maps * self.spatial_weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
        
        # 类别 gating
        gate = torch.sigmoid(self.spatial_gate(class_vec)).view(N, 1, 1, 1)
        spatial_attn = self.sigmoid(spatial_attn) * gate
        
        out = x_ca * spatial_attn + x  # 最终残差
        return out
"""
