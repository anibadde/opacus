#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional, List

from torch.utils.data import DataLoader
from opacus.accountants import RDPAccountant
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers import DPOptimizer, DPDDPOptimizer
from opacus.accountants.rdp import get_noise_multiplier
from opacus.distributed import (  
    DifferentiallyPrivateDistributedDataParallel     
        as DPDDP
)
from opacus.data_loader import DPDataLoader
from torch import nn, optim
from torch.utils.data import DataLoader


class PrivacyEngine:
    def __init__(self, secure_mode=False):
        self.accountant = RDPAccountant()
        self.secure_mode = secure_mode  # TODO: actually support it

    def make_private(
        self,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: float,
        batch_first: bool = True,
        loss_reduction: str = "mean",
    ):
        distributed = type(module) is DPDDP

        # TODO: DP-Specific validation
        # TODO: either validate consistent dataset or do per-dataset accounting
        module = self._prepare_model(module, batch_first, loss_reduction)
        data_loader = self._prepare_data_loader(data_loader, distributed=distributed)

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        optimizer = self._prepare_optimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            distributed=distributed,
        )

        def accountant_hook(optim: DPOptimizer):
            # TODO: This works for Poisson for both single-node and distributed
            # The reason is that the sample rate is the same in both cases (but in distributed mode, each node samples among a subset of the data)
            self.accountant.step(
                noise_multiplier=optim.noise_multiplier,
                sample_rate=sample_rate * optim.accumulated_iterations,
            )

        optimizer.attach_step_hook(accountant_hook)

        return module, optimizer, data_loader

    def make_private_with_epsilon(
        self,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: float,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        alphas: Optional[List[float]] = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
    ):
        sample_rate = 1 / len(data_loader)

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                epochs=epochs,
                alphas=alphas,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            ),
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
        )

    def _prepare_model(
        self,
        module: nn.Module,
        batch_first: bool = True,
        loss_reduction: str = "mean",
    ) -> GradSampleModule:
        if isinstance(module, GradSampleModule):
            return module
        else:
            return GradSampleModule(
                module, batch_first=batch_first, loss_reduction=loss_reduction
            )

    def _prepare_optimizer(
        self,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed : bool = False,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            # TODO: lol rename optimizer optimizer optimizer
            optimizer = optimizer.optimizer

        if distributed:
            return DPDDPOptimizer(
                optimizer=optimizer,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                expected_batch_size=expected_batch_size,
                loss_reduction=loss_reduction,
            )
        else:
            return DPOptimizer(
                optimizer=optimizer,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                expected_batch_size=expected_batch_size,
                loss_reduction=loss_reduction,
            )

    def _prepare_data_loader(self, data_loader: DataLoader, distributed: bool) -> DataLoader:
        return DPDataLoader.from_data_loader(data_loader, distributed=distributed)

    # TODO: default delta value?
    def get_epsilon(self, delta, alphas=None):
        return self.accountant.get_privacy_spent(delta)[0]



class PrivacyEngineUnsafeKeepDataLoader(PrivacyEngine):
    def _prepare_data_loader(self, data_loader: DataLoader) -> DataLoader:
        return data_loader
