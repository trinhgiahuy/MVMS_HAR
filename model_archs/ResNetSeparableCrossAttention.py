import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# SeparableConv2D block with kernel and bias initialization
class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2D, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels
        )
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Initialize the kernel and bias
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity='relu')
        nn.init.constant_(self.depthwise.bias, 0.2)
        nn.init.constant_(self.pointwise.bias, 0.2)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Residual Block with SeparableConv2D
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResNetBlock, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1

        self.conv1 = SeparableConv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SeparableConv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust the identity path if downsampling or if input and output channels are different
        if downsample or in_channels != out_channels:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
        else:
            self.downsample_conv = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsample if necessary
        if self.downsample_conv is not None:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)

        out += identity
        out = F.relu(out)

        return out

# Cross-Attention Layer
class CrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads
        assert (
            self.head_dim * num_heads == in_channels
        ), "in_channels must be divisible by num_heads"

        self.query_proj = nn.Linear(in_channels, in_channels)
        self.key_proj = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        # x1 and x2 are feature maps from view1 and view2
        batch_size, channels, height, width = x1.size()

        # Flatten spatial dimensions
        x1 = x1.view(batch_size, channels, -1)  # [B, C, N]
        x2 = x2.view(batch_size, channels, -1)  # [B, C, N]

        # Transpose for multi-head attention
        x1 = x1.permute(0, 2, 1)  # [B, N, C]
        x2 = x2.permute(0, 2, 1)  # [B, N, C]

        # Linear projections
        Q = self.query_proj(x1)  # [B, N, C]
        K = self.key_proj(x2)    # [B, N, C]
        V = self.value_proj(x2)  # [B, N, C]

        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, N]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, H, N, D]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.in_channels)  # [B, N, C]

        # Final linear projection
        out = self.out_proj(out)  # [B, N, C]

        # Reshape back to [B, C, H, W]
        out = out.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width)

        return out

# ResNet-like network using SeparableConv2D and Cross-Attention for Multi-View Radar Data
class ResNetSeparableCrossAttention(nn.Module):
    def __init__(self, name, num_classes=6, dtype=torch.float32):
        super(ResNetSeparableCrossAttention, self).__init__()
        self.name = "ResNetSeparableCrossAttention" + name

        # Modality streams
        self.modalities = ['Doppler', 'Azimuth', 'Elevation']
        self.num_modalities = len(self.modalities)

        # Modality-specific encoders
        self.encoders = nn.ModuleDict()
        for modality in self.modalities:
            # Shared layers for both views within a modality
            conv1 = SeparableConv2D(1, 32, kernel_size=7, stride=2, padding=3)
            bn1 = nn.BatchNorm2d(32)
            layer1 = self._make_layer(32, 64)
            layer2 = self._make_layer(64, 128, downsample=True)
            layer3 = self._make_layer(128, 256, downsample=True)
            layer4 = self._make_layer(256, 512, downsample=True)

            # Cross-Attention Layer
            cross_attention = CrossAttention(in_channels=512)

            self.encoders[modality] = nn.ModuleDict({
                'conv1': conv1,
                'bn1': bn1,
                'layer1': layer1,
                'layer2': layer2,
                'layer3': layer3,
                'layer4': layer4,
                'cross_attention': cross_attention
            })

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(512 * self.num_modalities, num_classes)

    def _make_layer(self, in_channels, out_channels, downsample=False):
        return ResNetBlock(in_channels, out_channels, downsample=downsample)

    def forward(self, x):
        # x shape: [batch_size, 3, 2, 32, 256]
        modality_features = []

        for i, modality in enumerate(self.modalities):
            # Extract modality data
            # x[:, i, :, :, :] has shape [batch_size, 2, 32, 256]
            modality_data = x[:, i, :, :, :]  # [batch_size, 2, 32, 256]
            view1 = modality_data[:, 0, :, :].unsqueeze(1)  # [batch_size, 1, 32, 256]
            view2 = modality_data[:, 1, :, :].unsqueeze(1)  # [batch_size, 1, 32, 256]

            encoder = self.encoders[modality]

            # Process view 1
            out1 = encoder['conv1'](view1)
            out1 = encoder['bn1'](out1)
            out1 = F.relu(out1)
            out1 = encoder['layer1'](out1)
            out1 = encoder['layer2'](out1)
            out1 = encoder['layer3'](out1)
            out1 = encoder['layer4'](out1)

            # Process view 2 (using the same encoder)
            out2 = encoder['conv1'](view2)
            out2 = encoder['bn1'](out2)
            out2 = F.relu(out2)
            out2 = encoder['layer1'](out2)
            out2 = encoder['layer2'](out2)
            out2 = encoder['layer3'](out2)
            out2 = encoder['layer4'](out2)

            # Cross-Attention Layer
            # Let out1 attend to out2 and vice versa
            out1_attended = encoder['cross_attention'](out1, out2)
            out2_attended = encoder['cross_attention'](out2, out1)

            # Fuse the two attended outputs
            out_fused = out1_attended + out2_attended

            # Global Average Pooling
            out_fused = self.global_avg_pool(out_fused)
            out_fused = torch.flatten(out_fused, 1)  # [batch_size, 512]

            modality_features.append(out_fused)

        # Concatenate modality features
        out = torch.cat(modality_features, dim=1)  # [batch_size, 512 * num_modalities]

        # Final fully connected layer
        out = self.fc(out)  # [batch_size, num_classes]

        return out

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path
