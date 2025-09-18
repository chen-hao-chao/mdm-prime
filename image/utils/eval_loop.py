# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import os
import time
import torch
import logging
import datetime
import PIL.Image
import numpy as np

from argparse import Namespace
from pathlib import Path
from typing import Iterable

from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.utils import save_image
from utils import distributed_mode
from utils.solver import MDM_Prime_EulerSolver, MDM_Prime_CO_EulerSolver
from utils.utils import encoder, decoder, create_p_mask, convert_to_marginal_time
from utils.utils import super_encode, super_decode
from utils.wrapper import WrappedPrimeModel

logger = logging.getLogger(__name__)

def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
):
    MASK_TOKEN = args.base 
    PRINT_FREQUENCY = args.print_freq

    gc.collect()
    wrapped_encoder = lambda x, seq: encoder(x, args.base, args.target_length, seq)
    p_mask = create_p_mask(wrapped_encoder, L=args.seq_length, C=args.vocab_size, l=args.target_length, b=args.base, device=device) if args.carry_over else None
    marginalization_fn = lambda p: convert_to_marginal_time(p=p, masks=p_mask)
    model.train(False)
    model = WrappedPrimeModel(model, marginalization_fn, temperature_parameters=args.temperature_parameters)

    if args.carry_over:
        solver = MDM_Prime_CO_EulerSolver(
            model=model,
            schedule_n=args.schedule_n,
            vocabulary_size=args.base+1,
            use_corrector_step=args.use_corrector_step,
            corrector_parameters=args.corrector_parameters,
            use_timestep_scheduler=args.use_timestep_scheduler,
            timestep_parameters=args.timestep_parameters,
            use_uniform_mixture=args.use_uniform_mixture,
            mixture_classes=args.mixture_classes
        )
    else:
        solver = MDM_Prime_EulerSolver(
            model=model,
            schedule_n=args.schedule_n,
            vocabulary_size=args.base + 1,
            use_corrector_step=args.use_corrector_step,
            corrector_parameters=args.corrector_parameters,
            use_timestep_scheduler=args.use_timestep_scheduler,
            timestep_parameters=args.timestep_parameters,
            encoder=(lambda x: wrapped_encoder(x, args.seq_length))
        )

    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )
    is_metric = InceptionScore(normalize=True).to(
        device=device, non_blocking=True
    )

    num_synthetic = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        fid_metric.update(samples, real=True)

        if args.use_uniform_mixture:
            labels = torch.randint(args.mixture_classes, labels.shape).to(device)

        if num_synthetic < fid_samples:
            model.reset_nfe_counter()
            sample_shape = samples.shape
            B, c, H, W = sample_shape
            c = 2 if args.super_token else 3
            y_1 = (torch.zeros((B, c*args.target_length, H, W), dtype=torch.long, device=device) + MASK_TOKEN)
            y_0 = solver.sample(y_init=y_1, step_size=1.0 / args.nfe, verbose=True)

            if args.target_length != 1:
                x_0 = decoder(y_0, args.base, args.target_length, args.seq_length)
            if args.super_token:
                x_0 = super_decode(y_0)
            
            x_0 = x_0.to(torch.float32) / 255.0
            logger.info(
                f"{samples.shape[0]} samples generated in {model.get_nfe()} evaluations."
            )
            if num_synthetic + x_0.shape[0] > fid_samples:
                x_0 = x_0[: fid_samples - num_synthetic]
            fid_metric.update(x_0, real=False)
            is_metric.update(x_0)
            num_synthetic += x_0.shape[0]
            if not snapshots_saved and args.output_dir:
                save_image(x_0, fp=Path(args.output_dir) / "snapshots" / f"{epoch}_{data_iter_step}.png")
                snapshots_saved = True

            if args.output_dir:
                images_np = (x_0 * 255.0).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                image_dir = Path(args.output_dir) / "eval_samples" / str(epoch)
                os.makedirs(image_dir, exist_ok=True)
                image_path = image_dir / "{}_{}.npy".format(distributed_mode.get_rank(), data_iter_step)
                np.save(image_path, images_np)

        if data_iter_step % PRINT_FREQUENCY == 0:
            gc.collect()
            running_fid = fid_metric.compute()
            logger.info(f"Evaluating [{data_iter_step}/{len(data_loader)}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}")
        
    return {"fid": float(fid_metric.compute().detach().cpu()), "is": float((is_metric.compute()[0]).detach().cpu())}

def eval_model_offline(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    args: Namespace,
    split_index: int,
    split_size: int,
):
    MASK_TOKEN = args.base
    
    wrapped_encoder = lambda x, seq: encoder(x, args.base, args.target_length, seq)
    p_mask = create_p_mask(wrapped_encoder, L=args.seq_length, C=args.vocab_size, l=args.target_length, b=args.base, device=device) if args.carry_over else None
    marginalization_fn = lambda p: convert_to_marginal_time(p=p, masks=p_mask)
    model.train(False)
    model = WrappedPrimeModel(model, marginalization_fn, temperature_parameters=args.temperature_parameters)

    if args.carry_over:
        solver = MDM_Prime_CO_EulerSolver(
            model=model,
            schedule_n=args.schedule_n,
            vocabulary_size=args.base+1,
            use_corrector_step=args.use_corrector_step,
            corrector_parameters=args.corrector_parameters,
            use_timestep_scheduler=args.use_timestep_scheduler,
            timestep_parameters=args.timestep_parameters,
            use_uniform_mixture=args.use_uniform_mixture,
            mixture_classes=args.mixture_classes
        )
    else:
        solver = MDM_Prime_EulerSolver(
            model=model,
            schedule_n=args.schedule_n,
            vocabulary_size=args.base + 1,
            use_corrector_step=args.use_corrector_step,
            corrector_parameters=args.corrector_parameters,
            use_timestep_scheduler=args.use_timestep_scheduler,
            timestep_parameters=args.timestep_parameters,
            encoder=(lambda x: wrapped_encoder(x, args.seq_length))
        )

    for data_iter_step, (samples, _) in enumerate(data_loader):
        if (data_iter_step * samples.shape[0] < split_size * split_index):
            continue
        elif data_iter_step * samples.shape[0] >= split_size * (split_index + 1):
            break
        else:
            start_time = time.time()
            logger.info(
                "Start generating samples: {} / {}".format(
                    data_iter_step * samples.shape[0] - split_size * split_index,
                    split_size,
                )
            )
            samples = samples.to(device, non_blocking=True)
            model.reset_nfe_counter()
            sample_shape = samples.shape
            B, c, H, W = sample_shape
            c = 2 if args.super_token else 3
            y_1 = (torch.zeros((B, c*args.target_length, H, W), dtype=torch.long, device=device) + MASK_TOKEN)
            y_0 = solver.sample(y_init=y_1, step_size=1.0 / args.nfe, verbose=True)
                
            if args.target_length != 1:
                x_0 = decoder(y_0, args.base, args.target_length, args.seq_length)
            if args.super_token:
                x_0 = super_decode(y_0)

            if args.output_dir:
                images_np = x_0.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                temperature_setup = ("temp={}_temp_n={}_temp_b={}".format(args.temperature_parameters[0], args.temperature_parameters[1], args.temperature_parameters[2]))
                corrector_setup = "" if (not args.use_corrector_step) else "_cor={}_cor_n={}_cor_b={}".format(args.corrector_parameters[0], args.corrector_parameters[1], args.corrector_parameters[2])
                timestep_setup = "" if (not args.use_timestep_scheduler) else "_ts={}_ts_n={}_ts_b={}".format(args.timestep_parameters[0], args.timestep_parameters[1], args.timestep_parameters[2])
                sampler_setup = temperature_setup + corrector_setup + timestep_setup
                sampling_nfe = "_step=" + str(args.nfe)
                if args.sampling_all:
                    dir_name = str(args.seed)
                    image_dir = Path(args.output_dir) / "fid_samples" / "samples_all" / (sampler_setup + sampling_nfe) / dir_name
                else:
                    dir_name = sampler_setup
                    image_dir = Path(args.output_dir) / "fid_samples" / dir_name
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / "{}.npy".format(data_iter_step)
                np.save(image_path, images_np)
                image_vis_path = image_dir / "{}.png".format(data_iter_step)
                PIL.Image.fromarray(images_np[0], "RGB").save(image_vis_path)

            x_0 = x_0.to(torch.float32) / 255.0
            logger.info(f"{samples.shape[0]} samples generated in {model.get_nfe()} evaluations.")

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))

            logger.info(f"Evaluation time {total_time_str}.")
            logger.info("-" * 10)

def calculate_fidis(
    data_loader: Iterable,
    device: torch.device,
    image_dir,
    subdir=None
):
    fid_metric = FrechetInceptionDistance(normalize=True).to(device=device)
    is_metric = InceptionScore(normalize=True).to(device=device)

    start_time = time.time()
    if subdir is not None:
        files = [f for sd in subdir if (p := image_dir / str(sd)).exists() for f in p.glob("*.npy") if f.is_file()]
    else:
        files = [x for x in (image_dir).rglob("*.npy") if x.is_file()]

    for image_path in files:
        print("Fake Samples: {}".format(image_path))
        images_np = np.load(image_path)
        synthetic_samples = torch.tensor(images_np).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        fid_metric.update(synthetic_samples, real=False)
        is_metric.update(synthetic_samples)

    for data_iter_step, (samples, _) in enumerate(data_loader):
        print("Real Samples: {} / {}".format(data_iter_step * samples.shape[0], 50000))
        samples = samples.to(device)
        fid_metric.update(samples, real=True)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Evaluation time {total_time_str}.")

    return {"fid": float(fid_metric.compute().detach().cpu()), "is": float((is_metric.compute()[0]).detach().cpu())}

def calculate_nll_elbo(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    args: Namespace,
    split_index: int,
    split_size: int,
    save_file: bool = True
):
    gc.collect()
    MASK_TOKEN = args.base

    model.train(False)
    
    # Time discretization
    n_discretization = 512
    discretization = (
        torch.linspace(0, 1, n_discretization + 1, device=device)[:-1]
        .view(-1, 1)
        .repeat(1, args.batch_size)
    )

    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    acc_nll = None
    acc_bdp = None
    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if (data_iter_step * samples.shape[0] < split_size * split_index):
            continue
        elif data_iter_step * samples.shape[0] >= split_size * (split_index + 1):
            break
        else:
            with torch.no_grad():
                start_time = time.time()
                print("Samples: {} / {}".format(data_iter_step * samples.shape[0], 50000))

                samples = samples.to(device, non_blocking=True)
                samples = (samples * 255.0).to(torch.long)
                if args.super_token:
                    samples = super_encode(samples)
                    
                # Lower variance estimator for time discretization
                discretization = discretization + torch.rand(size=(1, args.batch_size), device=device)
                discretization = discretization % 1
                discretization = discretization * (1 - args.epsilon)

                # Sample probability path
                B, c, H, W = samples.shape
                y_1 = torch.zeros((B, args.target_length * c, H, W), dtype=torch.long, device=device) + MASK_TOKEN
                y_0 = encoder(samples, args.base, args.target_length, c*H*W)

                if args.use_uniform_mixture:
                    uniform_id = torch.randint(args.mixture_classes, labels.shape).to(device)
                    conditioning = {"label": uniform_id}
                else:
                    conditioning = {}
                
                elbo = torch.zeros(size=(args.batch_size,), device=device)
                for t in discretization:
                    sigma_t = (1 - t**args.schedule_n).view(B, 1, 1, 1).expand(B, args.target_length * c, H, W)
                    source_indices = torch.rand(size=y_0.shape, device=y_0.device) < sigma_t
                    y_t = torch.where(condition=source_indices, input=y_1, other=y_0)

                    if args.target_length != 1:
                        logits = model(y_t, t=t, extra=conditioning)
                        loss = torch.nn.functional.cross_entropy(
                            logits.reshape([-1, 256]), samples.reshape([-1]), reduction='sum'
                        ) / args.batch_size
                    else:
                        if args.super_token:
                            logits = model(y_t, t=t, extra=conditioning)
                            loss = torch.nn.functional.cross_entropy(
                                logits.reshape([-1, 4097]), samples.reshape([-1]), reduction='sum'
                            ) / args.batch_size
                        else:
                            logits = model(y_t, t=t, extra=conditioning)
                            loss = torch.nn.functional.cross_entropy(
                                logits.reshape([-1, 257]), samples.reshape([-1]), reduction='sum'
                            ) / args.batch_size

                    alpha_t = t ** args.schedule_n
                    d_alpha_t = args.schedule_n * (t ** (args.schedule_n - 1))
                    fac = d_alpha_t / (1 - alpha_t)
                    elbo += loss * fac
                elbo /= n_discretization

            if save_file:
                nll_dir = Path(args.output_dir) / "nll" / str(epoch)
                os.makedirs(nll_dir, exist_ok=True)
                nll_path = nll_dir / "{}.npy".format(data_iter_step)
                np.save(nll_path, elbo.detach().cpu().numpy())

            pbd = elbo / (np.log(2.) * args.seq_length)

            acc_nll = torch.cat([acc_nll, elbo], dim=0) if acc_nll is not None else elbo
            acc_bdp = torch.cat([acc_bdp, pbd], dim=0) if acc_bdp is not None else pbd

            # Calculate accumulated time
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(f"Evaluation time {total_time_str}. NLL (ELBO) {acc_nll.mean().item()}. BPD {acc_bdp.mean().item()}")
            print("-" * 10)
    return {"acc_bdp": float(acc_bdp.mean().detach().cpu())}


def imputation(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    args: Namespace,
):
    MASK_TOKEN = args.base
    gc.collect()

    assert args.target_length != 1

    wrapped_encoder = lambda x, seq: encoder(x, args.base, args.target_length, seq)
    p_mask = create_p_mask(wrapped_encoder, L=args.seq_length, C=args.vocab_size, l=args.target_length, b=args.base, device=device) if args.carry_over else None
    marginalization_fn = lambda p: convert_to_marginal_time(p=p, masks=p_mask)
    model.train(False)
    model = WrappedPrimeModel(model, marginalization_fn, temperature_parameters=args.temperature_parameters)

    if args.carry_over:
        solver = MDM_Prime_CO_EulerSolver(
            model=model,
            schedule_n=args.schedule_n,
            vocabulary_size=args.base+1,
            use_corrector_step=False,
            corrector_parameters=args.corrector_parameters,
            use_timestep_scheduler=args.use_timestep_scheduler,
            timestep_parameters=args.timestep_parameters,
            use_uniform_mixture=args.use_uniform_mixture,
            mixture_classes=args.mixture_classes
        )
    else:
        solver = MDM_Prime_EulerSolver(
            model=model,
            schedule_n=args.schedule_n,
            vocabulary_size=args.base + 1,
            use_corrector_step=False,
            corrector_parameters=args.corrector_parameters,
            use_timestep_scheduler=args.use_timestep_scheduler,
            timestep_parameters=args.timestep_parameters,
            encoder=(lambda x: wrapped_encoder(x, args.seq_length))
        )

    if args.output_dir:
        (Path(args.output_dir) / "imputation").mkdir(parents=True, exist_ok=True)

    total_samples = 0
    for data_iter_step, (samples, _) in enumerate(data_loader):
        start_time = time.time()
        logger.info("Start generating samples: {}".format(data_iter_step * samples.shape[0]))

        samples = samples.to(device)
        B, c, H, W = samples.shape
        samples = torch.floor(samples * 255).to(dtype=torch.long, device=device)
        samples_enc = encoder(samples, args.base, args.target_length, c*H*W)
        conditions = samples_enc.view(B, c, args.target_length, H, W)

        for k in range(samples.shape[0]):
            condition = conditions[k, :, args.subtoken_index, :, :]
            model.reset_nfe_counter()

            c = 2 if args.super_token else 3
            y_1 = (torch.zeros((c, args.target_length, H, W), dtype=torch.long, device=device) + MASK_TOKEN)
            y_1[:, args.subtoken_index, :, :] = condition
            y_1 = y_1.view(c*args.target_length, H, W).unsqueeze(0).repeat(B, 1, 1, 1)
            y_0 = solver.sample(y_init=y_1, step_size=1.0 / args.nfe, verbose=True)
                
            if args.target_length != 1:
                x_0 = decoder(y_0, args.base, args.target_length, args.seq_length)
            if args.super_token:
                x_0 = super_decode(y_0)

            if args.output_dir:
                image_dir = Path(args.output_dir) / "imputation"
                os.makedirs(image_dir, exist_ok=True)
                image_vis_path = image_dir / "{}.png".format(total_samples)
                x_0 = x_0.to(torch.float32) / 255.0
                save_image(x_0, fp=image_vis_path)

                sample_np = (samples.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
                image_vis_path = image_dir / "batch_{}.png".format(total_samples)
                PIL.Image.fromarray(sample_np[k], "RGB").save(image_vis_path)

            logger.info(f"{samples.shape[0]} samples generated in {model.get_nfe()} evaluations.")

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            
            logger.info(f"Evaluation time {total_time_str}.")
            total_samples = total_samples + 1

        if total_samples >= args.fid_samples:
            break