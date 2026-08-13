"""Exact WRN-28-2 substrate from CDMAD commit 7cd732b.

The apparently unused imports and rotation head are intentionally omitted/retained
only where they affect execution. In particular, BatchNorm2d deliberately ignores
its constructor's momentum and eps arguments, matching the released code.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv3x3(input_channels: int, output_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(input_channels, output_channels, 3, stride, 1, bias=False)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, channels: int, momentum: float = 1e-3, eps: float = 1e-3):
        super().__init__(channels)
        self.update_batch_stats = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.update_batch_stats:
            return super().forward(x)
        return nn.functional.batch_norm(
            x, None, None, self.weight, self.bias, True, self.momentum, self.eps
        )


def relu() -> nn.LeakyReLU:
    return nn.LeakyReLU(0.1)


class Residual(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int = 1,
        activate_before_residual: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        if activate_before_residual:
            self.pre_act = nn.Sequential(BatchNorm2d(input_channels), relu())
        else:
            self.pre_act = nn.Identity()
            layers.extend([BatchNorm2d(input_channels), relu()])
        layers.extend(
            [
                conv3x3(input_channels, output_channels, stride),
                BatchNorm2d(output_channels),
                relu(),
                conv3x3(output_channels, output_channels),
            ]
        )
        if stride >= 2 or input_channels != output_channels:
            self.identity = nn.Conv2d(
                input_channels, output_channels, 1, stride, bias=False
            )
        else:
            self.identity = nn.Identity()
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_act(x)
        return self.identity(x) + self.layer(x)


class WRN(nn.Module):
    """WRN-28-width with LeakyReLU(0.1), matching CDMAD's executed model."""

    def __init__(self, width: int, num_classes: int, rotation: bool = True):
        super().__init__()
        self.init_conv = conv3x3(3, 16)
        filters = [16, 16 * width, 32 * width, 64 * width]
        self.unit1 = nn.Sequential(
            Residual(filters[0], filters[1], activate_before_residual=True),
            *[Residual(filters[1], filters[1]) for _ in range(1, 4)],
        )
        self.unit2 = nn.Sequential(
            Residual(filters[1], filters[2], 2),
            *[Residual(filters[2], filters[2]) for _ in range(1, 4)],
        )
        self.unit3 = nn.Sequential(
            Residual(filters[2], filters[3], 2),
            *[Residual(filters[3], filters[3]) for _ in range(1, 4)],
        )
        self.unit4 = nn.Sequential(
            BatchNorm2d(filters[3]), relu(), nn.AdaptiveAvgPool2d(1)
        )
        self.output = nn.Linear(filters[3], num_classes)
        self.rotation = rotation
        if self.rotation:
            self.rot = nn.Linear(filters[3], 4)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(
        self, x: torch.Tensor, return_feature: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]:
        x = self.init_conv(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        features = self.unit4(x)
        logits = self.output(features.squeeze())
        rotation = self.rot(features.squeeze()) if self.rotation else 0
        if return_feature:
            return [logits, rotation, features]
        return logits, rotation

    def update_batch_stats(self, flag: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.update_batch_stats = flag
