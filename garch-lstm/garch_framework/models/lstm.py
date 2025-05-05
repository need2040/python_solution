import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseGARCHModel


class LSTMGARCHModel(BaseGARCHModel):
    def __init__(self, kernel_fn, hidden_dim: int = 16, w: float = 1.0):
        super().__init__()
        self.kernel_fn = kernel_fn  # GARCH kernel module
        self.hidden_dim = hidden_dim
        self.w = w

        # LSTM cell weights
        self.W_f = nn.Linear(2, hidden_dim)
        self.W_i = nn.Linear(2, hidden_dim)
        self.W_c = nn.Linear(2, hidden_dim)
        self.W_o = nn.Linear(2, 1)

    def forward(self, eps_seq):
        # eps_seq: [batch_size, input_len], corresponds to (ε_{t-k+1}, ..., ε_{t-1})
        batch_size = eps_seq.size(0)

        # Prepare x_t = [ε_{t−1}, σ²_{t−1}] over time steps
        # Initialize σ²_{t−1} as empirical var of ε_{t−1}
        sigma2_tm1 = eps_seq[:, -1:].pow(2)  # simple estimate

        c_t = torch.zeros(batch_size, self.hidden_dim, device=eps_seq.device)
        h_t = torch.zeros_like(c_t)

        for t in range(self.input_len):
            eps_tm1 = eps_seq[:, t].unsqueeze(1)
            x_t = torch.cat([eps_tm1, sigma2_tm1], dim=1)  # [batch_size, 2]

            f_t = torch.sigmoid(self.W_f(x_t))
            i_t = torch.sigmoid(self.W_i(x_t))
            c_hat_t = torch.tanh(self.W_c(x_t))
            c_t = f_t * c_t + i_t * c_hat_t

        # Compute GARCH output (independently)
        garch_out = self.kernel_fn(torch.cat([eps_seq, sigma2_tm1], dim=1))  # shape: [batch_size]

        # Compute LSTM gate value and final σ²_t
        o_t = self.W_o(torch.cat([eps_seq[:, -1:], sigma2_tm1], dim=1)).squeeze(1)
        sigma2_t = garch_out * (1 + self.w * torch.tanh(o_t * c_t[:, 0]))  # scalar c_t[:, 0]
        return sigma2_t
