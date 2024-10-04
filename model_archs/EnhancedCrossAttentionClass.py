import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Radar-Specific Convolutional Encoder
class RadarEncoder(nn.Module):
    def __init__(self, in_channels):
        super(RadarEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        return x  # Output shape: [batch_size, 512, H', W']

# Enhanced Cross-Attention Layer with Learnable Positional Encoding
class EnhancedCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super(EnhancedCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, in_channels, 1, 1))

        # Linear projections
        self.qkv_proj = nn.Linear(in_channels, in_channels * 3)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        # x1 and x2: [batch_size, channels, height, width]
        batch_size, channels, height, width = x1.size()
        N = height * width

        # Flatten spatial dimensions and add positional encoding
        x1 = x1 + self.pos_embedding
        x2 = x2 + self.pos_embedding

        x1 = x1.view(batch_size, channels, N).permute(0, 2, 1)  # [B, N, C]
        x2 = x2.view(batch_size, channels, N).permute(0, 2, 1)  # [B, N, C]

        # Concatenate x1 and x2
        x = torch.cat([x1, x2], dim=1)  # [B, 2N, C]

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # [B, 2N, 3C]
        qkv = qkv.reshape(batch_size, 2*N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, 2N, D]

        Q, K, V = qkv[0], qkv[1], qkv[2]  # Each: [B, H, 2N, D]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, 2N, 2N]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, H, 2N, D]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, 2*N, self.in_channels)  # [B, 2N, C]

        # Final linear projection
        out = self.out_proj(out)  # [B, 2N, C]

        # Separate back into x1 and x2
        out1 = out[:, :N, :].permute(0, 2, 1).view(batch_size, channels, height, width)
        out2 = out[:, N:, :].permute(0, 2, 1).view(batch_size, channels, height, width)

        # Fuse the outputs (e.g., via addition)
        out_fused = out1 + out2  # [batch_size, channels, height, width]

        return out_fused

# Inter-Modality Attention Layer
class InterModalityAttention(nn.Module):
    def __init__(self, in_channels, num_modalities, num_heads=8, dropout=0.1):
        super(InterModalityAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.num_modalities = num_modalities
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"

        # Linear projections
        self.qkv_proj = nn.Linear(in_channels, in_channels * 3)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, modality_features):
        # modality_features: List of [batch_size, channels] tensors
        x = torch.stack(modality_features, dim=1)  # [batch_size, num_modalities, channels]
        batch_size = x.size(0)

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # [batch_size, num_modalities, 3 * channels]
        qkv = qkv.reshape(batch_size, self.num_modalities, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, num_modalities, head_dim]

        Q, K, V = qkv[0], qkv[1], qkv[2]  # Each: [batch_size, num_heads, num_modalities, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, num_modalities, num_modalities]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [batch_size, num_heads, num_modalities, head_dim]

        # Concatenate heads
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_modalities, self.in_channels)  # [batch_size, num_modalities, channels]

        # Aggregate modality features
        fused_feature = out.mean(dim=1)  # [batch_size, channels]

        return fused_feature

# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, feat_view1, feat_view2, labels):
        # Compute pairwise distances
        distances = F.pairwise_distance(feat_view1, feat_view2)

        # Labels are 1 if same class (positive pair), 0 otherwise (negative pair)
        loss = torch.mean(
            (1 - labels) * torch.pow(distances, 2) +
            labels * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        )
        return loss

# Classification Head
class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Full Model
class EnhancedCrossAttentionClass(nn.Module):
    def __init__(self, name, num_classes=6, dtype=torch.float32):
        super(EnhancedCrossAttentionClass, self).__init__()
        self.name = "EnhancedCrossAttentionClass" + name
        self.dtype = dtype

        self.modalities = ['Doppler', 'Azimuth', 'Elevation']
        self.num_modalities = len(self.modalities)
        self.in_channels = 512  # Output channels from RadarEncoder

        # Radar Encoders
        self.encoders = nn.ModuleDict({
            modality: RadarEncoder(in_channels=1)
            for modality in self.modalities
        })

        # Cross-Attention for each modality
        self.cross_attentions = nn.ModuleDict({
            modality: EnhancedCrossAttention(in_channels=self.in_channels)
            for modality in self.modalities
        })

        # Inter-Modality Attention
        self.inter_modality_attention = InterModalityAttention(
            in_channels=self.in_channels,
            num_modalities=self.num_modalities
        )

        # Classification Head
        self.classifier = ClassificationHead(in_features=self.in_channels, num_classes=num_classes)

        # Contrastive Loss Function
        self.contrastive_loss_fn = ContrastiveLoss(margin=1.0)

    def forward(self, x, labels=None):
        # x shape: [batch_size, 3, 2, 32, 256]
        batch_size = x.size(0)
        modality_features = []
        contrastive_losses = []

        for i, modality in enumerate(self.modalities):
            modality_data = x[:, i, :, :, :]  # [batch_size, 2, 32, 256]
            view1 = modality_data[:, 0, :, :].unsqueeze(1)  # [batch_size, 1, 32, 256]
            view2 = modality_data[:, 1, :, :].unsqueeze(1)  # [batch_size, 1, 32, 256]

            # Encode views
            encoder = self.encoders[modality]
            feat_view1 = encoder(view1)  # [batch_size, channels, H', W']
            feat_view2 = encoder(view2)

            # Cross-Attention Fusion
            cross_attention = self.cross_attentions[modality]
            fused_feature = cross_attention(feat_view1, feat_view2)  # [batch_size, channels, H', W']

            # Global Average Pooling
            fused_feature = F.adaptive_avg_pool2d(fused_feature, (1, 1)).view(batch_size, -1)  # [batch_size, channels]

            modality_features.append(fused_feature)

            # Compute contrastive loss if labels are provided
            if labels is not None:
                feat_view1_flat = F.adaptive_avg_pool2d(feat_view1, (1, 1)).view(batch_size, -1)
                feat_view2_flat = F.adaptive_avg_pool2d(feat_view2, (1, 1)).view(batch_size, -1)
                # Labels are binary: 1 if same class, 0 otherwise
                contrastive_label = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().to(self.dtype)
                contrastive_loss = self.contrastive_loss_fn(feat_view1_flat, feat_view2_flat, contrastive_label)
                contrastive_losses.append(contrastive_loss)

        # Inter-Modality Fusion
        fused_feature = self.inter_modality_attention(modality_features)  # [batch_size, channels]

        # Classification
        out = self.classifier(fused_feature)  # [batch_size, num_classes]

        # Compute total contrastive loss
        if labels is not None:
            total_contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)
            return out, total_contrastive_loss
        else:
            return out

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path
