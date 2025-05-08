import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseGARCHModel



class GARCHLSTM(nn.Module):
    def __init__(self, garch_core, hidden_size=16, mode='gjrgarch', scale=1):
        super().__init__()
        self.garch_core = garch_core
        
        # Параметры ворот
        self.W_f = nn.Linear(1, hidden_size)  # для ε_{t-1}
        self.U_f = nn.Linear(1, hidden_size)  # для σ²_{t-1}
        self.b_f = nn.Parameter(torch.zeros(1))

        self.W_i = nn.Linear(1, hidden_size)
        self.U_i = nn.Linear(1, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(1))

        self.W_c = nn.Linear(1, hidden_size)
        self.U_c = nn.Linear(1, hidden_size)
        self.b_c = nn.Parameter(torch.zeros(1))

        self.w = nn.Parameter(torch.tensor(0.5))
        self.hidden_size = hidden_size
        self.scale = scale
        self.mode = mode
    
    def init_hidden(self, batch_size=1):
        """Инициализация начального состояния памяти"""
        return torch.zeros(batch_size, self.hidden_size)
    
    def forward(self, eps_t_minus_1, sigma2_t_minus_1, c_t_minus_1=None):
        # Проверка и подготовка входных данных
        if eps_t_minus_1.dim() == 1:
            eps_t_minus_1 = eps_t_minus_1.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
        if sigma2_t_minus_1.dim() == 1:
            sigma2_t_minus_1 = sigma2_t_minus_1.unsqueeze(-1)
            
        batch_size = eps_t_minus_1.size(0)
        
        # Инициализация скрытого состояния если None
        if c_t_minus_1 is None:
            c_t_minus_1 = self.init_hidden(batch_size).to(eps_t_minus_1.device)
        elif c_t_minus_1.size(0) != batch_size:
            # Если размер батча изменился, инициализируем заново
            c_t_minus_1 = self.init_hidden(batch_size).to(eps_t_minus_1.device)

        # Вычисление компонентов LSTM
        f_t = torch.sigmoid(
            self.W_f(eps_t_minus_1) +  # [batch_size, hidden_size]
            self.U_f(sigma2_t_minus_1) +  # [batch_size, hidden_size]
            self.b_f.unsqueeze(0).expand(batch_size, -1)  # [hidden_size] -> [batch_size, hidden_size]
        )
        
        i_t = torch.sigmoid(
            self.W_i(eps_t_minus_1) + 
            self.U_i(sigma2_t_minus_1) + 
            self.b_i.unsqueeze(0).expand(batch_size, -1)
        )
        
        c_hat_t = torch.sigmoid(
            self.W_c(eps_t_minus_1) + 
            self.U_c(sigma2_t_minus_1) + 
            self.b_c.unsqueeze(0).expand(batch_size, -1)
        )

        # Обновление состояния ячейки
        c_t = f_t * c_t_minus_1 + i_t * c_hat_t

        # Подготовка входа для GARCH ядра
        eps2 = eps_t_minus_1 ** 2
        indicator = (eps_t_minus_1 < 0).float()
        asymm = eps2 * indicator
        omega_const = torch.ones_like(eps2) * self.scale
        garch_input = torch.cat([omega_const, eps2, asymm, sigma2_t_minus_1], dim=1)

        # Вычисление выхода GARCH
        o_t = self.garch_core(garch_input).unsqueeze(1)  # [batch_size, 1]

        # Финальный выход
        sigma2_t = o_t * (1 + self.w * torch.tanh(c_t).mean(dim=1, keepdim=True))

        return sigma2_t, c_t