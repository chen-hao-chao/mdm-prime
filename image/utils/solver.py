import torch
from math import ceil
from tqdm import tqdm
from torch import Tensor
from torch.nn import functional as F

# Source: https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/categorical_sampler.py
def categorical(probs: Tensor):
    return torch.multinomial(probs.flatten(0, -2), 1, replacement=True).view(
        *probs.shape[:-1]
    )

class MDM_Prime_CO_EulerSolver:
    def __init__(self, model, schedule_n, vocabulary_size, 
                 use_corrector_step, corrector_parameters,
                 use_timestep_scheduler, timestep_parameters,
                 use_uniform_mixture=False, mixture_classes=4):
        self.model = model
        self.schedule_n = schedule_n
        self.vocabulary_size = vocabulary_size

        if use_corrector_step:
            self.corrector = lambda t: corrector_parameters[0] * torch.pow(t, corrector_parameters[1]) * torch.pow(1.0 - t, corrector_parameters[1])
            print("corrector_parameters: ", corrector_parameters)
        else:
            self.corrector = None
        
        if use_timestep_scheduler:
            self.timestep_scheduler = lambda t: 1 - timestep_parameters[0] * (1-t)**timestep_parameters[1] + timestep_parameters[2]
            print("timestep_parameters: ", timestep_parameters)
        else:
            self.timestep_scheduler = lambda t: t
        
        self.use_uniform_mixture = use_uniform_mixture
        self.mixture_classes = mixture_classes

    @torch.no_grad()
    def sample(self, y_init: Tensor, step_size: float, **model_extras) -> Tensor:
        # Discretize timesteps
        t_init = 0.0
        t_final = 1.0
        n_steps = ceil((t_final - t_init) / step_size)
        t_discretization = torch.tensor(
            [t_init + step_size * i for i in range(n_steps)] + [t_final],
            device=y_init.device,
        )

        y_t = y_init.clone()
        steps_counter = 0
        ctx = tqdm(total=t_final, desc=f"NFE: {steps_counter}")
        with ctx:
            for i in range(n_steps):
                if i == n_steps - 1:
                    t = self.timestep_scheduler(t_discretization[i : i + 1])
                    if self.use_uniform_mixture:
                        y_t_expand = y_t.repeat(self.mixture_classes, 1, 1, 1)
                        t_expand = t.repeat(y_t_expand.shape[0])
                        uniform_label = torch.arange(self.mixture_classes).repeat_interleave(y_t.shape[0]).to(y_t.device)
                        p_0t = self.model(input=y_t_expand, t=t_expand, label=uniform_label).to(dtype=torch.float32)
                        p_0t = p_0t.view(self.mixture_classes, y_t.shape[0], *p_0t.shape[1:]).mean(dim=0)
                    else:
                        p_0t = self.model(input=y_t, t=t.repeat(y_t.shape[0]), **model_extras).to(dtype=torch.float32)
                    y_0 = categorical(p_0t)
                    y_1 = torch.ones(y_init.shape, dtype=y_init.dtype).to(y_init.device)*(self.vocabulary_size-1)
                    y_t[y_t == y_1] = y_0[y_t == y_1]

                else:
                    t = self.timestep_scheduler(t_discretization[i : i + 1])
                    s = self.timestep_scheduler(t_discretization[i + 1 : i + 2])
                    h = s - t

                    alpha_t = t**self.schedule_n
                    alpha_s = s**self.schedule_n

                    # masked distribution
                    y_1 = torch.ones(y_init.shape, dtype=y_init.dtype).to(y_init.device)*(self.vocabulary_size-1)
                    delta_1 = F.one_hot(y_1, num_classes=self.vocabulary_size).to(alpha_t.dtype)

                    # unmasked distribution
                    if self.use_uniform_mixture:
                        y_t_expand = y_t.repeat(self.mixture_classes, 1, 1, 1)
                        t_expand = t.repeat(y_t_expand.shape[0])
                        uniform_label = torch.arange(self.mixture_classes).repeat_interleave(y_t.shape[0]).to(y_t.device)
                        p_0t = self.model(input=y_t_expand, t=t_expand, label=uniform_label).to(dtype=torch.float32)
                        p_0t = p_0t.view(self.mixture_classes, y_t.shape[0], *p_0t.shape[1:]).mean(dim=0)
                    else:
                        p_0t = self.model(input=y_t, t=t.repeat(y_t.shape[0]), **model_extras).to(dtype=torch.float32)
                    
                    p_1t_comp = (alpha_s - alpha_t) / (1 - alpha_t) * p_0t
                    p_0t_comp = (1 - alpha_s) / (1 - alpha_t) * delta_1

                    y_s = categorical(p_0t_comp + p_1t_comp)
                    y_t[y_t == y_1] = y_s[y_t == y_1]

                    if self.corrector is not None:
                        d_alpha_t = self.schedule_n * (t ** (self.schedule_n - 1))
                        delta_0 = F.one_hot(categorical(p_0t), num_classes=self.vocabulary_size).to(alpha_t.dtype)
                        delta_t = F.one_hot(y_t, num_classes=self.vocabulary_size)
                        u = d_alpha_t / (1 - alpha_t) * delta_0 + \
                            self.corrector(t) * d_alpha_t / (alpha_t * (1 - alpha_t)) * ((1 - alpha_t) * delta_1 + alpha_t * delta_0)                        
                        u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)
                        mask_jump = torch.rand(size=y_t.shape, device=y_t.device) < 1 - torch.exp(-h * u.sum(dim=-1))
                        if mask_jump.sum() > 0:
                            y_t[mask_jump] = categorical(u[mask_jump].to(dtype=torch.float32))

                steps_counter += 1
                t = t + h

                ctx.n = t.item()
                ctx.refresh()
                ctx.set_description(f"NFE: {steps_counter}")
        return y_t
    
class MDM_Prime_EulerSolver:
    def __init__(self, model, schedule_n, vocabulary_size, 
                 use_corrector_step, corrector_parameters,
                 use_timestep_scheduler, timestep_parameters, encoder):
        super().__init__()
        self.model = model
        self.schedule_n = schedule_n
        self.vocabulary_size = vocabulary_size
        self.encoder = encoder

        if use_corrector_step:
            self.corrector = lambda t: corrector_parameters[0] * torch.pow(t, corrector_parameters[1]) * torch.pow(1.0 - t, corrector_parameters[1])
            print("corrector_parameters: ", corrector_parameters)
        else:
            self.corrector = None
        
        if use_timestep_scheduler:
            self.timestep_scheduler = lambda t: 1 - timestep_parameters[0] * (1-t)**timestep_parameters[1] + timestep_parameters[2]
            print("timestep_parameters: ", timestep_parameters)
        else:
            self.timestep_scheduler = lambda t: t

    @torch.no_grad()
    def sample(self, y_init: Tensor, step_size: float, **model_extras) -> Tensor:
        # Discretize timesteps
        t_init = 0.0
        t_final = 1.0
        n_steps = ceil((t_final - t_init) / step_size)
        t_discretization = torch.tensor(
            [t_init + step_size * i for i in range(n_steps)] + [t_final],
            device=y_init.device,
        )

        y_t = y_init.clone()
        steps_counter = 0
        ctx = tqdm(total=t_final, desc=f"NFE: {steps_counter}")
        with ctx:
            for i in range(n_steps):
                if i == n_steps - 1:
                    t = self.timestep_scheduler(t_discretization[i : i + 1])
                    p_0t = self.model(input=y_t, t=t.repeat(y_t.shape[0]), **model_extras)
                    y_0 = self.encoder(categorical(p_0t))
                    y_t[y_t == self.vocabulary_size-1] = y_0[y_t == self.vocabulary_size-1]
                else:
                    t = self.timestep_scheduler(t_discretization[i : i + 1])
                    s = self.timestep_scheduler(t_discretization[i + 1 : i + 2])
                    h = s - t

                    alpha_t = t**self.schedule_n
                    alpha_s = s**self.schedule_n

                    y_1 = torch.ones(y_init.shape, dtype=y_init.dtype).to(y_init.device)*(self.vocabulary_size-1)
                    delta_1 = F.one_hot(y_1, num_classes=self.vocabulary_size).to(alpha_t.dtype)
                    
                    p_0t = self.model(input=y_t, t=t.repeat(y_t.shape[0]), **model_extras).to(dtype=torch.float32)
                    y_0 = self.encoder(categorical(p_0t))
                    delta_0 = F.one_hot(y_0, num_classes=self.vocabulary_size).to(alpha_t.dtype)

                    p_m_comp = (1 - alpha_s) / (1 - alpha_t) * delta_1
                    p_0t_comp = (alpha_s - alpha_t) / (1 - alpha_t) * delta_0
                    
                    y_s = categorical(p_0t_comp + p_m_comp)
                    y_t[y_t == y_1] = y_s[y_t == y_1]

                    if self.corrector is not None:
                        d_alpha_t = self.schedule_n * (t ** (self.schedule_n - 1))
                        delta_t = F.one_hot(y_t, num_classes=self.vocabulary_size)
                        u = d_alpha_t / (1 - alpha_t) * delta_0 + \
                            self.corrector(t) * d_alpha_t / (alpha_t * (1 - alpha_t)) * ((1 - alpha_t) * delta_1 + alpha_t * delta_0)
                        u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)
                        mask_jump = torch.rand(size=y_t.shape, device=y_t.device) < 1 - torch.exp(-h * u.sum(dim=-1))
                        if mask_jump.sum() > 0:
                            y_t[mask_jump] = categorical(u[mask_jump].to(dtype=torch.float32))

                steps_counter += 1
                t = t + h

                ctx.n = t.item()
                ctx.refresh()
                ctx.set_description(f"NFE: {steps_counter}")

        return y_t
