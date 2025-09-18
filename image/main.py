# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
import time
import math
import json
import torch
import wandb
import logging
import datetime
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

from pathlib import Path
from arg_parser import get_args_parser
from models.model_configs import instantiate_model, instantiate_prime_model
from utils.dataloader import ImageNet32Dataset
from utils import distributed_mode
from utils.data_transform import get_train_transform
from utils.eval_loop import eval_model, calculate_nll_elbo
from utils.load_and_save import load_model, save_model
from utils.train_loop import train_one_epoch
from utils.grad_scaler import NativeScalerWithGradNormCount as NativeScaler

logger = logging.getLogger(__name__)

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))

    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)
        
        if args.wandb:
            wandb.login(key=args.wandb_key)
            if args.wandb_id != "":
                run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    config=args,
                    name=str(args.output_dir),
                    resume="allow",
                    id=args.wandb_id
                )
            else:  
                run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    config=args,
                    name=str(args.output_dir)
                )

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    if args.dataset == "imagenet":
        transform_train = get_train_transform()
        dataset_train = ImageNet32Dataset(args.data_path, transform=transform_train, split="train")
        dataset_val = ImageNet32Dataset(args.data_path, transform=transform_train, split="val")
    elif args.dataset == "cifar10":
        transform_train = get_train_transform()
        dataset_train = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    logger.info("Intializing DataLoader...")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))

    if args.dataset == "imagenet":
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    # Setup the base
    args.base = math.ceil(args.vocab_size ** (1/args.target_length))
    logger.info("Base: {}".format(str(args.base)))

    # Define the model
    logger.info("Initializing Model")
    if args.target_length != 1:
        model = instantiate_prime_model(
            use_ema=args.use_ema,
            target_length=args.target_length,
            base=args.base,
            vocab_size=args.vocab_size
        )
    else:
        model = instantiate_model(
            use_ema=args.use_ema,
            super_token=args.super_token
        )

    model.to(device)

    model_without_ddp = model
    logger.info(str(model_without_ddp))

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )

    lr_schedule = torch.optim.lr_scheduler.ConstantLR(
        optimizer, total_iters=args.epochs, factor=1.0
    )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            args=args,
        )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0):
            save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)
                if args.dataset == "imagenet":
                    data_loader_val.sampler.set_epoch(0)

            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                fid_samples = args.fid_samples // num_tasks
            
            with torch.no_grad():
                # Sampling
                eval_stats = eval_model(
                    model,
                    (data_loader_train if args.dataset != "imagenet" else data_loader_val),
                    device,
                    epoch=epoch,
                    fid_samples=fid_samples,
                    args=args,
                )
                log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})
                
                # Evaluating ELBO
                eval_stats_nll = calculate_nll_elbo(
                    model,
                    (data_loader_train if args.dataset != "imagenet" else data_loader_val),
                    device,
                    epoch=epoch,
                    args=args,
                    split_index=0,
                    split_size=fid_samples // 8,
                    save_file=False
                )
                log_stats.update({f"eval_{k}": v for k, v in eval_stats_nll.items()})

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
            if args.wandb:
                wandb.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    
    if args.wandb:
        run.finish()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
