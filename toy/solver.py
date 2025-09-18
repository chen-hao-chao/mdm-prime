from math import ceil

import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm


# Source: https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/categorical_sampler.py
def categorical(probs: Tensor):
    return torch.multinomial(probs.flatten(0, -2), 1, replacement=True).view(
        *probs.shape[:-1]
    )


class MDM_Prime_EulerSolver:
    def __init__(self, model, vocabulary_size, encoder, temperature=1.0):
        self.model = model
        self.vocabulary_size = vocabulary_size
        self.encoder = encoder
        self.temperature = temperature

    @torch.no_grad()
    def sample(self, y_init, step_size: float = 0.01, eps = 1e-12):
        # Discretize timesteps
        t_init = 1.0
        t_final = 1e-3
        n_steps = ceil((t_init - t_final) / step_size)
        t_discretization = torch.tensor([t_init - step_size * i for i in range(n_steps)] + [t_final], device=y_init.device)

        # Initialize the sampling process
        y_t = y_init.clone()
        steps_counter = 0
        ctx = tqdm(total=(t_init - t_final), desc=f"NFE: {steps_counter}")
        mask_token_id = self.vocabulary_size-1
        
        with ctx:
            for i in range(n_steps):
                t = t_discretization[i]
                s = t - step_size
                alpha_t = 1 - t
                alpha_s = 1 - s

                logits = self.model(x=y_t, t=t.repeat(y_t.shape[0]))
                p_0t = torch.softmax(logits / self.temperature, dim=-1)
                
                x_0 = categorical(p_0t)
                y_0 = self.encoder(x_0, p_0t.shape[1])

                if i == n_steps - 1:
                    y_t[y_t == mask_token_id] = y_0[y_t == mask_token_id]
                else:
                    is_mask = (y_t == mask_token_id)
                    p_unmask = ((alpha_s - alpha_t) / (1 - alpha_t + eps)).expand_as(y_t).to(dtype=torch.float32)
                    flip_to_y0 = (torch.rand_like(y_t, dtype=torch.float32) < p_unmask) & is_mask
                    y_t[flip_to_y0] = y_0[flip_to_y0]

                steps_counter += 1
                ctx.n = (1-t).item()
                ctx.refresh()
                ctx.set_description(f"NFE: {steps_counter}")

        return y_t
