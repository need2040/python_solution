import torch
from torch.utils.data import Dataset
import numpy as np

class FIGARCHDataset(Dataset):
    def __init__(self, eps, vol, truncation_size, scale=100):
        super().__init__()
        self.eps_squared = torch.tensor(np.square(eps) * scale).float()
        self.vol = torch.tensor(vol * scale).float()
        self.truncation_size = truncation_size

    def __len__(self):
        return len(self.vol) - self.truncation_size

    def __getitem__(self, idx):
        # Окно квадратов остатков и соответствующая волатильность
        return (
            self.eps_squared[idx:idx+self.truncation_size],  # [truncation_size]
            self.vol[idx+self.truncation_size]  # Скаляр
        )
    

class GJR_GARCHDataset(Dataset):
    def __init__(self, residuals, volatility, scale=100):
        self.scale = scale

        self.residuals = torch.tensor(residuals).float()
        self.volatility = torch.tensor(volatility).float()

        self.residuals_scaled = torch.square(self.residuals) * scale
        self.volatility_scaled = self.volatility * scale 
        

        self.inputs = torch.column_stack([
            torch.tensor(torch.ones_like(self.residuals_scaled)).float(),  
            torch.tensor(self.residuals_scaled).float(),
            torch.tensor(torch.where(self.residuals < 0,1,0) * self.residuals_scaled).float(),
            torch.tensor(self.volatility_scaled).float()
        ])
        

        self.outputs = torch.tensor(self.volatility_scaled).float()
        


    def __len__(self):
        return len(self.inputs)-1 
    
    def __getitem__(self, index):
        return (
            self.inputs[index],
            self.outputs[index+1],
            self.residuals_scaled[index+1]

        )
    
class GARCHDataset(Dataset):
    def __init__(self, residuals, volatility, scale=100):
        self.scale = scale

        self.residuals = torch.tensor(residuals).float()
        self.volatility = torch.tensor(volatility).float()

        self.residuals_scaled = torch.square(self.residuals) * scale
        self.volatility_scaled = self.volatility * scale 
        

        self.inputs = torch.column_stack([
            torch.tensor(torch.ones_like(self.residuals_scaled)).float(),  
            torch.tensor(self.residuals_scaled).float(),
            torch.tensor(self.volatility_scaled).float()
        ])
        

        self.outputs = torch.tensor(self.volatility_scaled).float()
        


    def __len__(self):
        return len(self.inputs)-1 
    
    def __getitem__(self, index):
        return (
            self.inputs[index],
            self.outputs[index+1], #Может быть полезной при условии 
            self.residuals_scaled[index+1]

        )
