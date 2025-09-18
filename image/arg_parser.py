# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging

logger = logging.getLogger(__name__)

def get_args_parser():
    parser = argparse.ArgumentParser("Image dataset training", add_help=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    # Training parameters
    parser.add_argument("--epochs", default=8001, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--num_gpus", default=8, type=int)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N",
        help="start epoch (used when resumed from checkpoint)",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--accum_iter", default=1, type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--eval_frequency", default=250, type=int,
        help="Frequency (in number of epochs) for running FID evaluation. -1 to never run evaluation.",
    )
    parser.add_argument(
        "--output_dir", default="/app/output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--fid_samples", default=2000, type=int,
        help="number of synthetic samples for FID evaluations",
    )
    # Optimizer parameters
    parser.add_argument(
        "--lr", type=float, default=0.0001,
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--optimizer_betas", nargs="+", type=float, default=[0.9, 0.95],
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--use_ema", action="store_true",
        help="When evaluating, use the model Exponential Moving Average weights.",
    )
    # Dataset parameters
    parser.add_argument(
        "--dataset", default="cifar10", type=str,
        help="Dataset to use.",
    )
    parser.add_argument(
        "--data_path", default="./data/image_generation", type=str,
        help="imagenet root folder with train, val and test subfolders",
    )
    # Distributed training parameters
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument(
        "--pin_mem", action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--mode", default="eval_fid", type=str)
    # Sampling parameters
    parser.add_argument("--schedule_n", type=float, default=3.0)
    parser.add_argument("--nfe", default=512, type=int)    
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--temperature_parameters", nargs="*", type=float, default=[1.0, 0.0, 0.0])
    parser.add_argument("--use_corrector_step", action='store_true')
    parser.add_argument("--corrector_parameters", nargs="*", type=float, default=[])
    parser.add_argument("--use_timestep_scheduler", action='store_true')
    parser.add_argument("--timestep_parameters", nargs="*", type=float, default=[])
    parser.add_argument("--sampling_all", action='store_true')
    parser.add_argument("--eval_all", action='store_true')
    parser.add_argument("--split_index", default=0, type=int)
    parser.add_argument("--split_size", default=6250, type=int)
    parser.add_argument("--sample_list", type=list, default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--subtoken_index', nargs='+', type=int, default=(0))
    # Wandb parameters
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str, default="mdm-prime")
    parser.add_argument("--wandb_id", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_key", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")
    # Ray parameters
    parser.add_argument("--storage_path", type=str, default="/app/results_ray")
    # Other parameters
    parser.add_argument("--target_length", type=int, default=1)
    parser.add_argument("--base", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=3072)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--carry_over", action='store_true')
    parser.add_argument("--super_token", action='store_true')
    parser.add_argument("--use_uniform_mixture", action='store_true')
    parser.add_argument("--mixture_classes", type=int, default=4)
    # Huggingface parameters
    parser.add_argument("--from_huggingface", action='store_true')
    parser.add_argument("--cache_dir", type=str, default="/app/huggingface_cache")

    return parser
