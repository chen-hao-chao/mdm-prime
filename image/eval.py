# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import math
import wandb
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

from pathlib import Path
from arg_parser import get_args_parser
from utils.data_transform import get_eval_transform
from utils.eval_loop import eval_model_offline, calculate_fidis, calculate_nll_elbo, imputation
from utils.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from utils.load_and_save import load_model
from utils.dataloader import ImageNet32Dataset
from models.model_configs import instantiate_model, instantiate_prime_model

logger = logging.getLogger(__name__)

def set_deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

def load_val_dataset(args):
    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_eval_transform()
    if args.dataset == "imagenet":
        dataset_train = ImageNet32Dataset(args.data_path, transform=transform_train, split="val")
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")
    return dataset_train

def init_model(args):
    logger.info(f"Initializing Model...")
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
    return model

def sample_data(args):
    device = torch.device(args.device)
    set_deterministic(args.seed)
    # Setup dataset
    dataset = load_val_dataset(args)
    data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True)
    # Setup model
    args.base = math.ceil(args.vocab_size ** (1/args.target_length))
    model = init_model(args)
    model.to(device)
    logger.info(str(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=args.optimizer_betas)
    lr_schedule = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=args.epochs, factor=1.0)
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")
    load_model(
        args=args,
        model_without_ddp=model,
        optimizer=optimizer,
        loss_scaler=NativeScaler(),
        lr_schedule=lr_schedule,
    )

    with torch.no_grad():
        eval_model_offline(
            model,
            data_loader_train,
            device,
            args=args,
            split_size=args.split_size,
            split_index=args.split_index,
        )

def sample_data_imputation(args):
    device = torch.device(args.device)
    set_deterministic(args.seed)
    # Setup dataset
    dataset = load_val_dataset(args)
    data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True)
    # Setup model
    args.base = math.ceil(args.vocab_size ** (1/args.target_length))
    model = init_model(args)
    model.to(device)
    logger.info(str(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=args.optimizer_betas)
    lr_schedule = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=args.epochs, factor=1.0)
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")
    load_model(
        args=args,
        model_without_ddp=model,
        optimizer=optimizer,
        loss_scaler=NativeScaler(),
        lr_schedule=lr_schedule,
    )
    with torch.no_grad():
        imputation(model, data_loader_train, device, args=args)

def compute_nll_bpd(args):
    device = torch.device(args.device)
    set_deterministic(args.seed)
    # Setup dataset
    dataset = load_val_dataset(args)
    data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True)
    # Setup model
    args.base = math.ceil(args.vocab_size ** (1/args.target_length))
    model = init_model(args)
    model.to(device)
    logger.info(str(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=args.optimizer_betas)
    lr_schedule = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=args.epochs, factor=1.0)
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")
    load_model(
        args=args,
        model_without_ddp=model,
        optimizer=optimizer,
        loss_scaler=NativeScaler(),
        lr_schedule=lr_schedule,
    )

    with torch.no_grad():
        eval_stats = calculate_nll_elbo(
            model,
            data_loader_train,
            device,
            args=args,
            epoch=args.start_epoch,
            split_size=args.split_size,
            split_index=args.split_index,
        )
    logger.info({f"eval_{k}": v for k, v in eval_stats.items()})

def eval_fid(args):
    set_deterministic(args.seed)
    device = torch.device(args.device)
    # Setup dataset
    dataset = load_val_dataset(args)
    data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True)
    with torch.no_grad():
        temperature_setup = ("temp={}_temp_n={}_temp_b={}".format(args.temperature_parameters[0], args.temperature_parameters[1], args.temperature_parameters[2]))
        corrector_setup = "" if (not args.use_corrector_step) else "_cor={}_cor_n={}_cor_b={}".format(args.corrector_parameters[0], args.corrector_parameters[1], args.corrector_parameters[2])
        timestep_setup = "" if (not args.use_timestep_scheduler) else "_ts={}_ts_n={}_ts_b={}".format(args.timestep_parameters[0], args.timestep_parameters[1], args.timestep_parameters[2])
        sampler_setup = temperature_setup + corrector_setup + timestep_setup
        sampling_nfe = "_step=" + str(args.nfe)
        if args.sampling_all:
            dir_name = str(args.seed)
            image_dir = Path(args.output_dir) / "fid_samples" / "samples_all" / (sampler_setup + sampling_nfe) / str(args.seed)
            result = calculate_fidis(data_loader_train, device, image_dir, subdir=None)
        elif args.eval_all:
            dir_name = args.output_dir.split('/')[-1]
            image_dir = Path(args.output_dir) / "fid_samples" / "samples_all" / (sampler_setup + sampling_nfe)
            result = calculate_fidis(data_loader_train, device, image_dir, subdir=args.sample_list)
        else:
            dir_name = sampler_setup
            image_dir = Path(args.output_dir) / "fid_samples" / dir_name
            result = calculate_fidis(data_loader_train, device, image_dir, subdir=None)
    print("FID: {} || IS: {}".format(result["fid"], result["is"]))

    if args.wandb:
        wandb.login(key=args.wandb_key)
        log_stats = {"epoch": args.start_epoch, "fid": result["fid"], "is": result["is"]}
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name=dir_name,
        )
        wandb.log(log_stats)
        run.finish()

def eval_nll(args):
    # Iterate through all files in the directory
    acc_nll = None
    acc_bdp = None
    nll_dir = Path(args.output_dir) / "nll" / str(args.start_epoch)
    for filename in os.listdir(nll_dir):
        file_path = os.path.join(nll_dir, filename)
        
        # Check if it is a file (not a directory)
        if os.path.isfile(file_path) and filename.endswith('.npy'):
            # Open the file in read mode
            elbo = np.load(file_path)
            print(f"Loaded .npy file: {filename}")
            pbd = elbo / (np.log(2.) * args.seq_length)
            acc_nll = np.concatenate([acc_nll, elbo], axis=0) if acc_nll is not None else elbo
            acc_bdp = np.concatenate([acc_bdp, pbd], axis=0) if acc_bdp is not None else pbd

    print(f"NLL (ELBO) {acc_nll.mean().item()}. BPD {acc_bdp.mean().item()}")
    if args.wandb:
        log_stats = {"epoch": args.start_epoch, "nll": acc_nll.mean().item(), "bpd": acc_bdp.mean().item()}
        wandb.login(key=args.wandb_key)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name=str(args.output_dir.split('/')[-1]),
        )
        wandb.log(log_stats)
        run.finish()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.mode == "sample_data":
        sample_data(args)
    elif args.mode == "eval_fid":
        eval_fid(args)
    elif args.mode == "compute_nll_bpd":
        compute_nll_bpd(args)
    elif args.mode == "eval_nll":
        eval_nll(args)
    elif args.mode == "imputation":
        sample_data_imputation(args)
    else:
        print("Unknown mode {}".format(args.mode))
        raise NotImplementedError