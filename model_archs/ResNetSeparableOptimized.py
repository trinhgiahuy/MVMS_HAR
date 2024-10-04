import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# SeparableConv2D block with kernel and bias initialization
class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2D, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
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

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = torch.mean(x, dim=[2, 3])  # Global average pooling
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)  # Rescale
        return x * y

# Residual Block with SeparableConv2D and SEBlock
class ResNetBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResNetBlockSE, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        
        self.conv1 = SeparableConv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SeparableConv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation Block
        self.se = SEBlock(out_channels)

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

        # Apply SE block
        out = self.se(out)

        # Apply downsample if necessary
        if self.downsample_conv is not None:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)

        out += identity
        out = F.relu(out)

        return out

# Self-Attention Mechanism for View Fusion
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(in_channels, in_channels)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        # Flatten the spatial dimensions (H * W) and permute
        x1_flat = x1.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        x2_flat = x2.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        # Project the queries, keys, and values
        Q = self.query(x1_flat)  # [B, H*W, C]
        K = self.key(x2_flat)    # [B, H*W, C]
        V = self.value(x2_flat)  # [B, H*W, C]

        # Split the queries, keys, and values into multiple heads
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]

        # Compute scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, num_heads, H*W, H*W]
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Multiply the attention weights by the values
        out = torch.matmul(attention_weights, V)  # [B, num_heads, H*W, head_dim]

        # Concatenate the multiple heads
        out = out.transpose(1, 2).contiguous().view(B, H * W, C)  # [B, H*W, C]

        # Final linear projection to match input shape
        out = self.fc_out(out)

        # Reshape back to original [B, C, H, W]
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return out



# Optimized ResNet Separable with Self-Attention and SEBlock
class ResNetSeparableOptimized(nn.Module):
    def __init__(self, name, num_classes=6, dtype=torch.float32):
        super(ResNetSeparableOptimized, self).__init__()
        self.name = "ResNetSeparableOptimized_" + name

        # First view stream
        self.conv1_view1 = SeparableConv2D(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1_view1 = nn.BatchNorm2d(32)
        self.layer1_view1 = self._make_layer(32, 64)
        self.layer2_view1 = self._make_layer(64, 128, downsample=True)
        self.layer3_view1 = self._make_layer(128, 256, downsample=True)
        self.layer4_view1 = self._make_layer(256, 512, downsample=True)

        # Second view stream
        self.conv1_view2 = SeparableConv2D(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1_view2 = nn.BatchNorm2d(32)
        self.layer1_view2 = self._make_layer(32, 64)
        self.layer2_view2 = self._make_layer(64, 128, downsample=True)
        self.layer3_view2 = self._make_layer(128, 256, downsample=True)
        self.layer4_view2 = self._make_layer(256, 512, downsample=True)

        # Self-Attention for fusion
        self.attention = MultiHeadSelfAttention(512, num_heads=8)

        # Global Average Pooling for both views
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, downsample=False):
        return ResNetBlockSE(in_channels, out_channels, downsample=downsample)

    def forward(self, x):
        # Split into two views
        view1 = x[:, :, 0, :, :]  # [batch_size, 3, 32, 256]
        view2 = x[:, :, 1, :, :]  # [batch_size, 3, 32, 256]

        # Process view 1
        out1 = self.conv1_view1(view1)
        out1 = self.bn1_view1(out1)
        out1 = self.layer1_view1(out1)
        out1 = self.layer2_view1(out1)
        out1 = self.layer3_view1(out1)
        out1 = self.layer4_view1(out1)

        # Process view 2
        out2 = self.conv1_view2(view2)
        out2 = self.bn1_view2(out2)
        out2 = self.layer1_view2(out2)
        out2 = self.layer2_view2(out2)
        out2 = self.layer3_view2(out2)
        out2 = self.layer4_view2(out2)

        # Self-Attention Fusion
        fused = self.attention(out1, out2)

        # Global Average Pooling
        fused = self.global_avg_pool(fused)
        fused = torch.flatten(fused, 1)  # [batch_size, 512]

        # Final fully connected layer
        out = self.fc(fused)
        return out

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path
