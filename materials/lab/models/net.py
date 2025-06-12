import torch
import torch.nn as nn
from copy import deepcopy
import layer

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

import torch
import torch.nn as nn
from copy import deepcopy
from .. import layer

class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, HW, C]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_flat + attn_out
        x = x + self.mlp(self.norm2(x))
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            #conv.append(SEBlock(channels))  # 加入SE注意力
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
    

def test_models():
    import torch
    import neuron, surrogate, functional

    x = torch.rand([2, 2, 128, 128])
    net = DVSGestureNet(16, neuron.IFNode, surrogate_function=surrogate.ATan())
    print(net(x).shape)
    functional.reset_net(net)
    functional.set_step_mode(net, 'm')
    x = torch.rand([4, 2, 2, 128, 128])
    print(net(x).shape)
    functional.reset_net(net)
    del net
    del x

if __name__ == "__main__":
    test_models()
    print("Test passed.")
