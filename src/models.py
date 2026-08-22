import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# =====================================================================
# 1. TriNet (Siamese Metric Learning Model)
# =====================================================================

class TriNetModel(nn.Module):
    """
    Standard Siamese Backbone with Projection Head for Triplet Loss.
    Supports DenseNet121, MobileNetV3, and EfficientNet backbones.
    """
    def __init__(self, backbone_name='densenet121', embedding_dim=128, pretrained=True):
        super(TriNetModel, self).__init__()
        self.backbone_name = backbone_name
        
        # Select and adapt the backbone
        if backbone_name == 'densenet121':
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            base = models.densenet121(weights=weights)
            in_features = base.classifier.in_features
            base.classifier = nn.Identity()
            self.backbone = base
            
        elif backbone_name == 'mobilenet_v3':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            in_features = base.classifier[0].in_features
            base.classifier = nn.Identity()
            self.backbone = base

        elif backbone_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Projection head to output normalized low-dimensional embedding
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        # Handle cases where feature maps aren't pooled automatically
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
            
        embeddings = self.projection_head(features)
        # L2-normalize embeddings for cosine / Euclidean distance comparisons
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings


# =====================================================================
# 2. PCB (Part-based Convolutional Baseline)
# =====================================================================

class PCBModel(nn.Module):
    """
    Part-based Convolutional Baseline.
    Slices final feature map into K horizontal parts (e.g., 6 horizontal stripes).
    """
    def __init__(self, num_classes=81, num_parts=6, hidden_dim=256, pretrained=True):
        super(PCBModel, self).__init__()
        self.num_parts = num_parts
        
        # Base feature extractor using DenseNet121 features block
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        base = models.densenet121(weights=weights)
        self.features = base.features
        in_channels = 1024  # DenseNet121 out channels before pooling

        # Local Conv layers for each stripe (1x1 conv to reduce dimensions)
        self.part_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_parts)
        ])

        # Classification heads for each part (for PCB loss training)
        self.part_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_parts)
        ])

    def forward(self, x):
        # Extract 2D feature map: [B, 1024, H, W]
        feat_map = self.features(x)
        H = feat_map.size(2)
        stripe_h = H // self.num_parts

        part_features = []
        part_logits = []

        for i in range(self.num_parts):
            # Extract slice along the height dimension (subcarrier/spatial domain)
            stripe = feat_map[:, :, i * stripe_h : (i + 1) * stripe_h, :]
            # Average pool slice to [B, 1024, 1, 1]
            pooled = F.adaptive_avg_pool2d(stripe, (1, 1))
            # Reduce dimension with 1x1 conv -> [B, 256, 1, 1]
            reduced = self.part_convs[i](pooled).view(pooled.size(0), -1)
            # Normalize part feature
            norm_part = F.normalize(reduced, p=2, dim=1)
            
            part_features.append(norm_part)
            part_logits.append(self.part_classifiers[i](reduced))

        if self.training:
            # During training, return part logits list for PCBLoss calculation
            return part_logits
        else:
            # During evaluation, concatenate all part features into a single descriptor
            return torch.cat(part_features, dim=1)


# =====================================================================
# 3. DuATM (Dual-Attention Matching Network Module)
# =====================================================================

class SpatialChannelAttention(nn.Module):
    """Dual Attention block (Channel + Spatial Attention)"""
    def __init__(self, in_channels, reduction=16):
        super(SpatialChannelAttention, self).__init__()
        # Channel Attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_gate(x)
        x_ca = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_gate(sa_input)
        
        return x_ca * sa


class DuATMModel(nn.Module):
    """
    Dual-Attention Matching Model for Re-ID.
    Applies Spatial + Channel Attention before feature pooling.
    """
    def __init__(self, embedding_dim=128, pretrained=True):
        super(DuATMModel, self).__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        base = models.densenet121(weights=weights)
        
        self.features = base.features
        in_channels = 1024
        
        # Dual Attention Module
        self.attention = SpatialChannelAttention(in_channels)
        
        # Output Embedding Layer
        self.embedding_head = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        # 1. Base Conv Features
        feat_map = self.features(x)
        # 2. Refine via Dual Attention
        attended_map = self.attention(feat_map)
        # 3. Global Pooling
        pooled = F.adaptive_avg_pool2d(attended_map, (1, 1)).view(attended_map.size(0), -1)
        # 4. Project & L2 Normalize
        embeddings = self.embedding_head(pooled)
        return F.normalize(embeddings, p=2, dim=1)