from __future__ import annotations
from .optimizer import DPOptimizer

from typing import Callable, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer



class DPPerLayerOptimizer(DPOptimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norms: List[float],
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
    ):
        self.max_grad_norms = max_grad_norms
        self.max_grad_norm = torch.sqrt(sum([c**2 for c in max_grad_norms]))
        super().__init__(optimizer, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm, expected_batch_size=expected_batch_size, loss_reduction=loss_reduction)

   
    def clip_and_accumulate(self):
        for (p, max_grad_norm) in zip(self.params, self.max_grad_norms):
            per_sample_norms = p.grad_sample.view(len(p.grad_sample), -1).norm(2, dim=-1)
            per_sample_clip_factor = (max_grad_norm / (per_sample_norms + 1e-6)).clamp(
                max=1.0
            )

            grad = torch.einsum("i,i...", per_sample_clip_factor, p.grad_sample)

            if hasattr(p, "summed_grad"):
                p.summed_grad += grad
            else:
                p.summed_grad = grad

    def add_noise(self):
        for p in self.params:
            noise = _generate_noise(
                self.noise_multiplier * self.max_grad_norm, p.summed_grad
            )
            p.grad = p.summed_grad + noise