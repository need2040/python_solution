import torch
import torch.nn as nn
from .base import BaseGARCHModel

class LSTMGARCH(BaseGARCHModel):
    def __init__(self, garch_type='GARCH', hidden_size=32, w=1.0, 
                 init_omega=0.2, init_alpha=0.2, init_beta=0.2, 
                 init_gamma=0.15, truncation_size=None):
        """
        LSTM-GARCH модель
        
        Args:
            garch_type (str): Тип GARCH модели ('GARCH', 'GJR-GARCH', 'FI-GARCH')
            hidden_size (int): Размер скрытого состояния LSTM
            w (float): Вес влияния LSTM на выход GARCH
            init_omega, init_alpha, init_beta, init_gamma: Параметры инициализации GARCH моделей
            truncation_size (int): Размер усечения для FI-GARCH
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.w = w
        self.garch_type = garch_type
        
        # Входной размер для LSTM (2: epsilon_{t-1} и sigma^2_{t-1})
        input_size = 2
        
        # Инициализация GARCH ядра
        if garch_type == 'GARCH':
            self.garch_kernel = nn.Sequential(
                nn.Linear(2, 1, bias=False)
            )
            init_weights = torch.tensor([init_alpha, init_beta], dtype=torch.float32)
            with torch.no_grad():
                self.garch_kernel[0].weight.copy_(init_weights.unsqueeze(0))
            self.omega = nn.Parameter(torch.tensor(init_omega, dtype=torch.float32))
            
        elif garch_type == 'GJR-GARCH':
            self.garch_kernel = nn.Sequential(
                nn.Linear(3, 1, bias=False)
            )
            init_weights = torch.tensor([init_alpha, init_gamma, init_beta], dtype=torch.float32)
            with torch.no_grad():
                self.garch_kernel[0].weight.copy_(init_weights.unsqueeze(0))
            self.omega = nn.Parameter(torch.tensor(init_omega, dtype=torch.float32))
            
        elif garch_type == 'FI-GARCH':
            if not truncation_size:
                raise ValueError("truncation_size must be specified for FI-GARCH")
            self.truncation_size = truncation_size
            self.garch_kernel = nn.Conv1d(
                in_channels=1, 
                out_channels=1, 
                kernel_size=self.truncation_size,
                bias=False
            )
            with torch.no_grad():
                self.garch_kernel.weight.data = torch.ones_like(self.garch_kernel.weight.data)
            self.omega = nn.Parameter(torch.tensor(init_omega, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown GARCH type: {garch_type}")
        
        # LSTM gates
        # Forget gate
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        
        # Input gate
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        
        # Output gate (replaced by GARCH kernel)
        
        # Cell state
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))
        
        # Sigmoid and tanh activations
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def _compute_garch_output(self, epsilon_t_1, sigma2_t_1):
        """Вычисление выхода GARCH ядра"""
        if self.garch_type == 'GARCH':
            inputs = torch.stack([epsilon_t_1, sigma2_t_1], dim=1)
            return self.omega + self.garch_kernel(inputs).squeeze(-1)
        
        elif self.garch_type == 'GJR-GARCH':
            # Для GJR-GARCH добавляем индикатор отрицательных epsilon
            I = (epsilon_t_1 < 0).float()
            inputs = torch.stack([epsilon_t_1**2, I * epsilon_t_1**2, sigma2_t_1], dim=1)
            return self.omega + self.garch_kernel(inputs).squeeze(-1)
        
        elif self.garch_type == 'FI-GARCH':
            # Для FI-GARCH предполагаем, что вход - это последовательность epsilon_{t-1} до t-truncation_size
            # и sigma2_{t-1} берется как последний элемент
            batch_size = epsilon_t_1.size(0)
            x = epsilon_t_1.view(batch_size, 1, -1)  # [batch, 1, truncation_size]
            conv_out = self.garch_kernel(x).squeeze(-1)  # [batch, 1]
            return self.omega + conv_out.squeeze(-1)
    
    def forward(self, epsilon_t_1, sigma2_t_1, h_t_1=None, c_t_1=None):
        """
        Forward pass
        
        Args:
            epsilon_t_1 (torch.Tensor): Ошибка на предыдущем шаге [batch_size]
            sigma2_t_1 (torch.Tensor): Дисперсия на предыдущем шаге [batch_size]
            h_t_1 (torch.Tensor): Скрытое состояние на предыдущем шаге [batch_size, hidden_size]
            c_t_1 (torch.Tensor): Состояние ячейки на предыдущем шаге [batch_size, hidden_size]
            
        Returns:
            sigma2_t (torch.Tensor): Дисперсия на текущем шаге [batch_size]
            h_t (torch.Tensor): Новое скрытое состояние [batch_size, hidden_size]
            c_t (torch.Tensor): Новое состояние ячейки [batch_size, hidden_size]
        """
        batch_size = epsilon_t_1.size(0)
        
        # Инициализация скрытых состояний, если они не предоставлены
        if h_t_1 is None:
            h_t_1 = torch.zeros(batch_size, self.hidden_size, device=epsilon_t_1.device)
        if c_t_1 is None:
            c_t_1 = torch.zeros(batch_size, self.hidden_size, device=epsilon_t_1.device)
        
        # Подготовка входных данных
        inputs = torch.stack([epsilon_t_1, sigma2_t_1], dim=1)
        
        # Forget gate
        f_t = self.sigmoid(self.W_f(inputs) + self.U_f(h_t_1) + self.b_f)
        
        # Input gate
        i_t = self.sigmoid(self.W_i(inputs) + self.U_i(h_t_1) + self.b_i)
        
        # GARCH kernel (output gate replacement)
        o_t = self._compute_garch_output(epsilon_t_1, sigma2_t_1)
        
        # Cell state
        c_hat_t = self.tanh(self.W_c(inputs) + self.U_c(h_t_1) + self.b_c)
        c_t = f_t * c_t_1 + i_t * c_hat_t
        
        # Вычисление дисперсии
        sigma2_t = o_t * (1 + self.w * self.tanh(c_t).sum(dim=1))  # sum over hidden dim
        
        # В LSTM-GARCH скрытое состояние не используется для выхода, но сохраняется для следующего шага
        h_t = o_t.unsqueeze(-1) * self.tanh(c_t)  # Аналогично оригинальному LSTM
        
        return sigma2_t, h_t, c_t