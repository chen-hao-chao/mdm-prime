import torch
from torch import nn

class LearnableSwish(nn.Module):
  def __init__(self, dim=-1):
    """
    Swish from: https://github.com/wgrathwohl/LSD/blob/master/networks.py#L299
    """
    super().__init__()
    self.beta = nn.Parameter(torch.ones((dim,)))

  def forward(self, x):
    return x * torch.sigmoid(self.beta[None, :] * x)

class MLPModel(nn.Module):
    def __init__(
        self, input_dim=(512,2), output_dim=(512,2), 
        time_dim=1,  hidden_dim=128, layers=2, 
        layernorm=False,  dropout_rate=None, 
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.time_embedding = nn.Linear(1, time_dim)
        self.token_embedding = torch.nn.Embedding(self.input_dim[0], hidden_dim)

        net = nn.ModuleList([])
        net.append(LearnableSwish(hidden_dim * self.input_dim[1] + time_dim))
        net.append(nn.Linear(hidden_dim * self.input_dim[1] + time_dim, hidden_dim))
        net.append(LearnableSwish(hidden_dim))
        
        for _ in range(layers):
            net.append(nn.Linear(hidden_dim, hidden_dim))
            if layernorm:
                net.append(nn.LayerNorm(hidden_dim))
            net.append(LearnableSwish(hidden_dim))
            if dropout_rate is not None:
                net.append(nn.Dropout(p=dropout_rate))
            
        net.append(nn.Linear(hidden_dim, self.output_dim[0] * self.output_dim[1]))
        self.main = nn.Sequential(*net)

    def forward(self, x, t):
        t = self.time_embedding(t.unsqueeze(-1))
        x = self.token_embedding(x)

        B, N, d = x.shape
        x = x.reshape(B, N * d)

        h = torch.cat([x, t], dim=1)
        h = self.main(h)
        h = h.reshape(B, self.output_dim[1], self.output_dim[0])

        return h