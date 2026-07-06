"""WideResNet for CIFAR (standard implementation)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(self.drop(F.relu(self.bn2(out))))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, num_classes=100, dropout=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        ch = [16, 16*k, 32*k, 64*k]
        self.conv = nn.Conv2d(3, ch[0], 3, 1, 1, bias=False)
        self.layer1 = self._make(n, ch[0], ch[1], 1, dropout)
        self.layer2 = self._make(n, ch[1], ch[2], 2, dropout)
        self.layer3 = self._make(n, ch[2], ch[3], 2, dropout)
        self.bn = nn.BatchNorm2d(ch[3])
        self.fc = nn.Linear(ch[3], num_classes)

    def _make(self, n, in_ch, out_ch, stride, dropout):
        layers = [_Block(in_ch, out_ch, stride, dropout)]
        for _ in range(n - 1):
            layers.append(_Block(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
