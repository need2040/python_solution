import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseGARCHModel


class GARCHLSTM(BaseGARCHModel):
    def __init__(self, garch_core, hidden_size=16):
        super().__init__()
        self.garch_core = garch_core  # объект класса GARCHModel / GJRGARCHModel / FIGARCHModel

        # LSTM "ядро памяти" на вход [ε_{t-1}, σ²_{t-1}]
        self.W_f = nn.Linear(2, hidden_size)
        self.U_f = nn.Linear(1, hidden_size, bias=False)

        self.W_i = nn.Linear(2, hidden_size)
        self.U_i = nn.Linear(1, hidden_size, bias=False)

        self.W_c = nn.Linear(2, hidden_size)
        self.U_c = nn.Linear(1, hidden_size, bias=False)

        self.o_gate_linear = nn.Linear(2, 1)  # для создания w ∈ [0, 1]
        self.w = nn.Parameter(torch.tensor(0.5))  # learnable controller

        self.hidden_size = hidden_size

    def forward(self, eps_t_minus_1, sigma2_t_minus_1, c_t_minus_1):
        """
        eps_t_minus_1: [batch_size, 1] — ε_{t-1}
        sigma2_t_minus_1: [batch_size, 1] — σ²_{t-1}
        c_t_minus_1: [batch_size, hidden_size] — память из прошлого шага
        """

        x = torch.cat([eps_t_minus_1, sigma2_t_minus_1], dim=1)  # [batch_size, 2]

        f_t = torch.sigmoid(self.W_f(x) + self.U_f(sigma2_t_minus_1))  # [batch_size, hidden]
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(sigma2_t_minus_1))
        c̃_t = torch.tanh(self.W_c(x) + self.U_c(sigma2_t_minus_1))

        c_t = f_t * c_t_minus_1 + i_t * c̃_t  # новое состояние памяти

        # GARCH-ядро: отдельно обучаемый модуль
        garch_input = torch.cat([eps_t_minus_1, sigma2_t_minus_1], dim=1)
        o_t = self.garch_core(garch_input).unsqueeze(1)  # [batch_size, 1]

        sigma2_t = o_t * (1 + self.w * torch.tanh(c_t).mean(dim=1, keepdim=True))  # [batch_size, 1]

        return sigma2_t, c_t
