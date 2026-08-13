"""Evaluation-model EMA with CDMAD's source-level quirks preserved."""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightEMA:
    def __init__(
        self,
        model: nn.Module,
        ema_model: nn.Module,
        learning_rate: float,
        decay_coefficient: float,
        alpha: float = 0.999,
    ):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = decay_coefficient * learning_rate

        # Direction intentionally matches CDMAD: fresh EMA state overwrites online state.
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    @torch.no_grad()
    def step(self) -> None:
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            # For integer buffers .float() creates a detached temporary, exactly as upstream.
            ema_float = ema_param.float()
            param_float = param.float()
            ema_float.mul_(self.alpha)
            ema_float.add_(param_float * one_minus_alpha)
            param_float.mul_(1 - self.wd)
