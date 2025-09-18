import os
import time
import numpy as np
import torch
import argparse
from dataclasses import dataclass
import math

from solver import MDM_Prime_EulerSolver
from model import MLPModel

from dataset import convert_img_to_array, img_sampler
from utils import decoder, encoder, visualize_img_samples, pngs_to_gif

def training(args, device):
    print("Start training")
    mask_token = args.base

    numpy_img = convert_img_to_array(args.image_path, args.vocab_size)
    numpy_img = numpy_img / numpy_img.sum()
    
    if args.resume_step == 0:
        probability_denoiser = MLPModel(
            input_dim=(args.base+1, args.target_length*args.seq_length), 
            output_dim=(args.vocab_size, args.seq_length),
            time_dim=1,
            hidden_dim=args.hidden_dim,
            layernorm=True,
            dropout_rate=0.1
        ).to(device)
    else:
        print("Resuming the training from the {}-th step.".format(args.resume_step))
        probability_denoiser = torch.load(os.path.join(args.load_path, "denoiser.pth"), weights_only=False)

    # init optimizer
    optim = torch.optim.Adam(probability_denoiser.parameters(), lr=args.lr)

    # train
    start_time = time.time()
    for i in range(args.resume_step+1, args.iterations):
        # Initialization
        optim.zero_grad()
        probability_denoiser.train()

        # Sample latent variables
        x_0 = img_sampler(numpy_img, size=args.vocab_size, batch_size=args.batch_size, device=device)
        bs = x_0.shape[0]
        dtype = x_0.dtype

        y_0 = encoder(x_0, args.base, args.target_length, args.seq_length)
        y_1 = (torch.zeros((bs, args.target_length * args.seq_length), device=device, dtype=dtype) + mask_token)
        
        t = torch.rand(bs).to(device)
        p_mask = (t).view(bs, 1).expand(bs, args.target_length * args.seq_length)
        mask_indices = torch.rand(size=y_0.shape, device=y_0.device) < p_mask
        y_t = torch.where(condition=mask_indices, input=y_1, other=y_0)

        # Calculate the loss function
        logits = probability_denoiser(x=y_t, t=t)
        log_p = torch.log_softmax(logits, dim=-1)
        log_p_0t = torch.gather(log_p, dim=-1, index=x_0.unsqueeze(-1))
        log_p_0t = log_p_0t.view(bs, args.seq_length)
        fac = (-1 / (t+args.epsilon)).view(bs, 1).expand(bs, args.seq_length)
        loss = torch.mean(log_p_0t * fac)

        # Optimizer step
        loss.backward()
        optim.step()

        # Print loss values and visualize learned distributions
        if (i + 1) % args.print_every == 0:
            elapsed = time.time() - start_time
            print("| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ".format(
                    i + 1, elapsed * 1000 / args.print_every, loss.item()
                )
            )
            torch.save(probability_denoiser, os.path.join(args.load_path, "denoiser_last.pth"))
            start_time = time.time()
            
        if (i + 1) % (args.print_every * 10) == 0:
            torch.save(probability_denoiser, os.path.join(args.load_path, "denoiser_{}.pth".format(i + 1)))
            eval_elbo(args=args, device=device, epoch=i + 1)
            sampling(args=args, device=device, epoch=i + 1)

def sampling(args, device, epoch=0):
    mask_token = args.base
    wrapped_encoder = lambda x, seq: encoder(x, args.base, args.target_length, seq)
    
    probability_denoiser = torch.load(os.path.join(args.load_path, "denoiser_{}.pth".format(epoch)), weights_only=False)
    probability_denoiser.eval()
    
    solver = MDM_Prime_EulerSolver(
        model=probability_denoiser,
        vocabulary_size=args.base + 1,
        encoder=wrapped_encoder,
        temperature=args.temperature,
    )

    step_size = 1 / args.nfe
    n_samples = 100000
    sample_iter = 5

    y_init = (torch.zeros(size=(n_samples, args.target_length * args.seq_length), device=device) + mask_token).long()

    samples = None
    for i in range(sample_iter):
        y_0 = solver.sample(y_init=y_init, step_size=step_size)
        sample = decoder(y_0, args.base, args.target_length, args.seq_length).cpu().numpy()
        samples = (
            np.concatenate((samples, sample), axis=0)
            if (samples is not None)
            else sample
        )

    file_name = "target_length=" + str(args.target_length) + "_nfe=" + str(args.nfe) + \
                "_temp=" + str(args.temperature) + "_epoch=" + str(epoch) + ".png"
    visualize_img_samples(samples, args.vocab_size, os.path.join(args.save_path, file_name))

def eval_elbo(args, device, epoch=0):
    print("Evaluate ELBO -- epoch: ", epoch)
    n_discretization = 128
    mask_token = args.base

    probability_denoiser = torch.load(os.path.join(args.load_path, "denoiser_{}.pth".format(epoch)), weights_only=False)
    probability_denoiser.eval()

    # Get data
    numpy_img = convert_img_to_array(args.image_path, args.vocab_size)
    numpy_img = numpy_img / numpy_img.sum()

    x_0 = img_sampler(numpy_img, size=args.vocab_size, batch_size=args.batch_size, device=device)
    bs = x_0.shape[0]
    dtype = x_0.dtype

    y_0 = encoder(x_0, args.base, args.target_length, args.seq_length)
    y_1 = (torch.zeros((bs, args.target_length * args.seq_length), device=device, dtype=dtype) + mask_token)

    # Time discretization
    discretization = (torch.linspace(0, 1, n_discretization + 1, device=device)[:-1].view(-1, 1).repeat(1, x_0.shape[0]))
    elbo = torch.zeros(size=(x_0.shape[0],), device=device)

    with torch.no_grad():
        # Lower variance estimator for time discretization
        # Source: https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_discrete_flow_matching.ipynb
        discretization = discretization + torch.rand(size=(1, x_0.shape[0]), device=device)
        discretization = discretization % 1
        discretization = discretization * (1 - args.epsilon)

        for t in discretization:
            # sample yt
            t = t.to(device)
            p_mask = (t).view(bs, 1).expand(bs, args.target_length * args.seq_length)
            mask_indices = torch.rand(size=y_0.shape, device=y_0.device) < p_mask
            y_t = torch.where(condition=mask_indices, input=y_1, other=y_0)

            logits = probability_denoiser(x=y_t, t=t)
            log_p = torch.log_softmax(logits, dim=-1)
            log_p_0t = torch.gather(log_p, dim=-1, index=x_0.unsqueeze(-1))
            log_p_0t = log_p_0t.view(bs, args.seq_length)
            fac = (- 1 / (t+args.epsilon)).view(bs, 1).expand(bs, args.seq_length)
            loss = (log_p_0t * fac).sum(-1)

            # compute ELBO
            elbo += loss

        elbo /= n_discretization

    elbo_print = ("ELBO (" + str(args.target_length) + "bit_" + str(args.base) + "base): " + str(elbo.mean().item()))
    print(elbo_print)

def set_deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

@dataclass
class Args:
    seed: int = 1
    target_length: int = 1
    base: int = 512
    mode: str = "train"
    lr: float = 0.001
    epsilon: float = 0.005
    batch_size: int = 4096
    iterations: int = 1000001
    print_every: int = 5000
    hidden_dim: int = 512
    vocab_size: int = 512
    seq_length: int = 2
    resume_step: int = 0
    load_path: str = ""
    save_path: str = ""
    image_path: str = "assets/cat_1.jpg"
    nfe: int = 128
    temperature: float = 1.0

def main(args=None):
    args = Args
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_length", type=int, default=1)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000001)
    parser.add_argument("--print_every", type=int, default=5000)
    parser.add_argument("--workdir", type=str, default="results")
    parser.add_argument("--rootdir", type=str, default="/app")
    parser.add_argument("--image_path", type=str, default="assets/cat_1.jpg")
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--nfe", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    args_ = parser.parse_args()

    args.target_length = args_.target_length
    args.base = math.ceil(args.vocab_size ** (1/args.target_length))
    args.mode = args_.mode
    args.resume_step = args_.resume_step
    args.print_every = args_.print_every
    args.image_path = args_.image_path
    args.nfe = args_.nfe
    args.iterations = args_.iterations
    args.temperature = args_.temperature

    file_path = "l={}_image={}".format(str(args.target_length), str(args.image_path.split("/")[-1].split(".")[0]))
    load_path = os.path.join(args_.rootdir, args_.workdir, file_path)
    args.load_path = load_path

    if torch.cuda.is_available():
        device = "cuda"
        print("Using gpu")
    else:
        device = "cpu"
        print("Using cpu.")

    # Make the training & sampling processes deterministic
    set_deterministic(0)

    if args.mode == "train":
        args.save_path = load_path
        if not os.path.exists(load_path):
            os.makedirs(load_path)
            print("Creating a new directory: ", load_path)
        print("Saving directory: ", args.save_path)
        training(args=args, device=device)
    elif args.mode == "sample":
        args.save_path = args_.save_path
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            print("Creating a new directory: ", args.save_path)
        print("Saving directory: ", args.save_path)
        sampling(args=args, device=device, epoch=args.resume_step)
    elif args.mode == "eval_elbo":
        eval_elbo(args=args, device=device, epoch=args.resume_step)
    else:
        print("unknown mode: ", args.mode)
        raise NotImplementedError

if __name__ == "__main__":
    main()
