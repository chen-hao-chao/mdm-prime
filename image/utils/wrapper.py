import torch

class WrappedPrimeModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, marginalize_fn, temperature_parameters=[1.0, 0.0, 0.0]):
        super().__init__()
        self.model = model
        self.marginalize_fn = marginalize_fn
        self.nfe_counter = 0
        self.temp_scheduler = lambda t: temperature_parameters[0] * t**temperature_parameters[1] + temperature_parameters[2]
        self.logit = None
        
    def forward(self, input: torch.Tensor, t: torch.Tensor, **extras):
        model_output = self.model(input, t, extra={})
        B, c, W, H, C = model_output.shape
        temp = self.temp_scheduler(t)[:, None, None, None, None].expand(-1, c, W, H, C)
        self.logit = model_output / temp
        p = torch.softmax(self.logit, dim=-1)
        out = self.marginalize_fn(p)
        self.nfe_counter += 1
        return out
    
    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter