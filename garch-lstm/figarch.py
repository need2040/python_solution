# %%
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from arch import arch_model
from pytorch_lightning.loggers import  TensorBoardLogger
from scipy.optimize import root
import matplotlib.pyplot as plt
import os 
from datetime import datetime
import json
from figarch_eval import fit_lambda_parameters, compute_lambda_sequence

# %%

# Генерация данных
def generate_ground_garch(omega, d, phi, beta, n=3000):
    am = arch_model(None, mean='Zero', vol='FIGARCH', p=1, q=1, power=2)
    params = np.array([omega, d, phi, beta])
    am_data = am.simulate(params, n)
    return am_data['data'].to_numpy(), am_data['volatility'].to_numpy()


# %%

# Dataset для полных последовательностей
class FullSequenceDataset(Dataset):
    def __init__(self, eps, vol, truncation_size, scale=100):
        super().__init__()
        self.eps_squared = torch.tensor(np.square(eps) * scale)
        self.vol = (torch.tensor(vol * scale))
        self.truncation_size = truncation_size
        self.eps_squared = self.eps_squared.float()
        self.vol = self.vol.float()

    def __len__(self):
        return len(self.eps_squared) - self.truncation_size  # Количество возможных окон

    def __getitem__(self, idx):
        # Возвращаем окно остатков и соответствующий таргет
        return (
            self.eps_squared[idx:idx+self.truncation_size].flip(-1),
            self.eps_squared[idx+self.truncation_size]
        )


# %%
class FIGARCHDM(pl.LightningDataModule):
    def __init__(self, eps, vol, truncation_size, batch_size):
        super().__init__()

        self.eps = eps
        self.vol = vol
        self.truncation_size = truncation_size
        self.batch_size = batch_size
    def setup(self, stage = None):
        self.train_dataset = FullSequenceDataset(self.eps, self.vol, self.truncation_size)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size= self.batch_size ,shuffle=False)

# %%
class CorrectedNLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_var, target_eps_squared):

        loss =  (0.5*(torch.log(pred_var)) + (target_eps_squared/(2*pred_var))) 
        return loss.mean()

# %%
class CorrectedTLoss(nn.Module):
    def __init__(self, nu):
        super().__init__()
        self.nu = nu
    def forward(self, pred_var, target_eps_squared):

        loss =  (0.5*(torch.log(pred_var))) + (0.5 * (self.nu+1))* torch.log(1 + (target_eps_squared/((self.nu-2)*pred_var)))
        return loss.mean()

# %%

# Модель FIGARCH с CNN
class FullConvFIGARCH(pl.LightningModule):
    def __init__(self, truncation_size, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.truncation_size = truncation_size
        self.lr = lr
        
        self.weights = nn.Parameter(torch.linspace(1.0, 0.1, truncation_size))
        
        self.loss_fn = CorrectedNLoss()

    def forward(self, x):
        weights = self.weights
        #weights = torch.nn.functional.softplus(self.weights)
        return torch.sum(x * weights, dim=1)

    def training_step(self, batch, batch_idx):
        eps_window, target_eps = batch
        pred_var = self.forward(eps_window)
        loss = self.loss_fn(pred_var, target_eps)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=20, 
            verbose=True
        )
        return {
            'optimizer': optimizer,
            #'lr_scheduler': {
            #    'scheduler': scheduler,
            #     'monitor': 'train_loss'
            #}
        }



# %%
# omega > 0 <- 1
# 0 <= d <= 1 <- 2
# 0 <= phi <= (1 - d) / 2 <- 2
# 0 <= beta <= d + phi <- 2


omega, d, phi, beta = 0.1, 0.5, 0.2, 0.3
data, volat = generate_ground_garch(omega, d, phi, beta)


# %%


# %%
ground_truth = [omega, d, phi, beta]

# %%
truncation_size = 30
#batch_size = (len(data) - truncation_size)//20
batch_size = 10

# %%
exact_lambdas = compute_lambda_sequence(d, phi, beta, 5)

# %%
exact_lambdas

# %%
model = FullConvFIGARCH(truncation_size)
dm = FIGARCHDM(data, volat, truncation_size, batch_size)


# %%
logger = TensorBoardLogger('tb_logs', 'figarch_model')

# %%
trainer = pl.Trainer(max_epochs=100, accelerator='auto', logger= logger)

# %%
trainer.fit(model, dm)

# %%
with torch.no_grad():
    actual_lambdas =  model.weights.data
    print("Learned weights:", actual_lambdas )

# %%
model_outputs = fit_lambda_parameters(actual_lambdas[:5])

# %%
model_outputs

# %%
model.weights.data

# %%
def compute_omega(weights, eps_squared, vol_series, trunc):
    """
    weights: learned lambdas (λ)
    eps_squared: εₜ² (квадраты остатков)
    vol_series: σₜ² (волатильность в квадрате)
    """
    weights = weights.numpy()
    truncation_size = len(weights[:trunc])
    
    
    pred = np.sum(weights[:truncation_size] * eps_squared[:truncation_size])
    

    omega = vol_series[truncation_size] - pred
    
    return max(omega, 1e-6)  

# %%
pred_omega = compute_omega(actual_lambdas, data, volat, 3)

# %%
pred_omega

# %%
def save_results(model_outputs, ground_truth, pred_omega, filename='figarch_results.json'):
    # Подготовка данных
    result = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'ground_truth': {
            'omega' : float(omega),
            'd': float(ground_truth[1]),
            'phi': float(ground_truth[2]),
            'beta': float(ground_truth[3]),
        },
        'model_params': {
            'omega' : float(pred_omega),
            'd': float(model_outputs[0]),
            'phi': float(model_outputs[1]),
            'beta': float(model_outputs[2])
        }

    }
    
    # Запись в файл
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode, encoding='utf-8') as f:
        f.write(json.dumps(result, indent=4) + '\n')  # Добавляем перевод строки

# %%
save_results(model_outputs, ground_truth, pred_omega)

# %%



