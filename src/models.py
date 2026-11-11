import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm 

# =====================================================================
# 1. TriNet (Siamese Metric Learning Model)
# =====================================================================

class TriNetModel(nn.Module):
    """
    Standard Siamese Backbone with Projection Head for Triplet Loss.
    Supports original backbones (DenseNet121, MobileNetV3, EfficientNet-B0)
    and NEW modern backbones (MobileNetV4, EfficientNetV2-S, ConvNeXt-Tiny).
    """
    def __init__(self, backbone_name='convnext_tiny', embedding_dim=128, pretrained=True):
        super(TriNetModel, self).__init__()
        self.backbone_name = backbone_name
        
        # --- Original Models ---
        if backbone_name == 'densenet121':
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            base = models.densenet121(weights=weights)
            base.classifier = nn.Identity()
            self.backbone = base
            
        elif backbone_name == 'mobilenet_v3':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            base.classifier = nn.Identity()
            self.backbone = base

        elif backbone_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            base.classifier = nn.Identity()
            self.backbone = base
            
        # --- NEW Modern Models ---
        elif backbone_name == 'mobilenet_v4':
            self.backbone = timm.create_model(
                'mobilenetv4_conv_small.e2400_r224_in1k', 
                pretrained=pretrained, 
                num_classes=0
            )
            
        elif backbone_name == 'efficientnet_v2_s':
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_v2_s(weights=weights)
            base.classifier = nn.Identity()
            self.backbone = base
            
        elif backbone_name == 'convnext_tiny':
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = models.convnext_tiny(weights=weights)
            base.classifier = nn.Identity()
            self.backbone = base
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Dynamically calculate the exact output dimension using a dummy tensor.
        # This prevents any mat1/mat2 shape mismatches across different CNN architectures.
        with torch.no_grad():
            # CHANGE: Batch size increased from 1 to 2 to satisfy BatchNorm
            dummy_input = torch.randn(2, 3, 224, 224) 
            dummy_out = self.backbone(dummy_input)
            if len(dummy_out.shape) > 2:
                dummy_out = F.adaptive_avg_pool2d(dummy_out, (1, 1)).view(dummy_out.size(0), -1)
            in_features = dummy_out.shape[1]

        # Projection head to output normalized low-dimensional embedding
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        
        # Handle cases where feature maps aren't pooled automatically
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
            
        embeddings = self.projection_head(features)
        
        # L2-normalize embeddings for cosine / Euclidean distance comparisons
        return F.normalize(embeddings, p=2, dim=1)