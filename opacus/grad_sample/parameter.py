#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .utils import create_or_extend_grad_sample, register_grad_sampler


@register_grad_sampler(nn.Parameter)
def compute_parameter_grad_sample(
    layer: nn.Parameter, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    """
    Computes per sample gradients for ``nn.Linear`` layer

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    gs = torch.einsum("n...i,n...j->nij", B, A)
    create_or_extend_grad_sample(layer.data, gs, batch_dim)