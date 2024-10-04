import torch
import torch.nn as nn
import os

# SeparableConv2D block for reducing computation
class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        print(f"Input to SeparableConv2D: {x.shape}")
        x = self.depthwise(x)
        x = self.pointwise(x)
        print(f"Output of SeparableConv2D: {x.shape}")
        return x

# Channel Attention Mechanism
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(f"Input to ChannelAttention: {x.shape}")
        avg_out = self.fc(self.avg_pool(x))
        print(f"Output of ChannelAttention: {avg_out.shape}")
        return self.sigmoid(avg_out)

# Corrected Spatial Attention Mechanism
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(32, 1, kernel_size=7, padding=3, bias=False)  # Adjusting input channels to 32
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(f"Input to SpatialAttention: {x.shape}")
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max
        x = torch.cat([avg_out, max_out], dim=1)  # Concatenate along channel axis
        x = self.conv1(x)
        print(f"Output of SpatialAttention: {x.shape}")
        return self.sigmoid(x)

# Simplified Multi-View ResNet with Attention
class SimpleMultiViewNet(nn.Module):
    def __init__(self, name, num_classes=6, dtype=torch.float32):
        super(SimpleMultiViewNet, self).__init__()
        self.name = "SimpleMultiViewNet" + name
        # First view stream
        self.conv1_view1 = SeparableConv2D(3, 32, kernel_size=5, stride=2, padding=1)
        self.bn1_view1 = nn.BatchNorm2d(32)
        
        # Second view stream
        self.conv1_view2 = SeparableConv2D(3, 32, kernel_size=5, stride=2, padding=1)
        self.bn1_view2 = nn.BatchNorm2d(32)

        # Attention mechanisms for each view
        self.channel_attention1 = ChannelAttention(32)
        self.spatial_attention1 = SpatialAttention()  # Spatial attention with correct input channels
        self.channel_attention2 = ChannelAttention(32)
        self.spatial_attention2 = SpatialAttention()  # Spatial attention with correct input channels

        # Global Average Pooling for both views
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(32 * 2, num_classes)

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Expecting [batch_size, 3, 2, 32, 256]

        # Split into two views
        view1 = x[:, :, 0, :, :]  # [batch_size, 3, 32, 256]
        view2 = x[:, :, 1, :, :]  # [batch_size, 3, 32, 256]
        
        # Process view 1
        out1 = self.conv1_view1(view1)
        out1 = self.bn1_view1(out1)
        out1 = self.channel_attention1(out1) * self.spatial_attention1(out1)
        print(f"After attention view1, shape: {out1.shape}")
        out1 = self.global_avg_pool(out1)
        out1 = torch.flatten(out1, 1)
        print(f"After global pooling view1, shape: {out1.shape}")

        # Process view 2
        out2 = self.conv1_view2(view2)
        out2 = self.bn1_view2(out2)
        out2 = self.channel_attention2(out2) * self.spatial_attention2(out2)
        print(f"After attention view2, shape: {out2.shape}")
        out2 = self.global_avg_pool(out2)
        out2 = torch.flatten(out2, 1)
        print(f"After global pooling view2, shape: {out2.shape}")

        # Concatenate both views
        out = torch.cat((out1, out2), dim=1)
        print(f"After concatenating both views, shape: {out.shape}")

        # Final fully connected layer
        out = self.fc(out)
        print(f"Final output shape: {out.shape}")

        return out

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path

# # Testing the model
# if __name__ == "__main__":
#     # Simulate input
#     batch_size = 64
#     input_data = torch.randn(batch_size, 3, 2, 32, 256)  # batch_size=64, 3 heatmaps, 2 views, size 32x256
    
#     model = SimpleMultiViewNet(name="TestModel")
#     output = model(input_data)
