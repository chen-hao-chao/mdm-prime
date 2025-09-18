# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable

import torch
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from utils.grad_scaler import NativeScalerWithGradNormCount
from utils.utils import encoder, super_encode

logger = logging.getLogger(__name__)

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    MASK_TOKEN = args.base
    PRINT_FREQUENCY = args.print_freq

    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)
    accum_iter = args.accum_iter
    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
        
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if args.use_uniform_mixture:
            uniform_id = torch.randint(args.mixture_classes, labels.shape).to(device)
            conditioning = {"label": uniform_id}
        else:
            conditioning = {}

        with torch.no_grad():
            samples = (samples * 255.0).to(torch.long)
            
            if args.super_token:
                samples = super_encode(samples)                

            # Sample latent variables
            B, c, H, W = samples.shape
            y_1 = torch.zeros((B, args.target_length * c, H, W), dtype=torch.long, device=device) + MASK_TOKEN
            y_0 = encoder(samples, args.base, args.target_length, c*H*W)
            t = torch.torch.rand(samples.shape[0]).to(device)
            sigma_t = (1 - t**args.schedule_n).view(B, 1, 1, 1).expand(B, args.target_length * c, H, W)
            source_indices = torch.rand(size=y_0.shape, device=y_0.device) < sigma_t
            y_t = torch.where(condition=source_indices, input=y_1, other=y_0)

        # Calculate the objective function
        if args.target_length != 1:
            logits = model(y_t, t=t, extra=conditioning)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape([-1, 256]), samples.reshape([-1])
            )
        else:
            if args.super_token:
                logits = model(y_t, t=t, extra=conditioning)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape([-1, 4097]), samples.reshape([-1])
                )
            else:
                logits = model(y_t, t=t, extra=conditioning)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape([-1, 257]), samples.reshape([-1])
                )

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
