from .models import GARCHModel, FIGARCHModel, GJRGARCHModel, GARCHLSTM
from typing import List,Tuple, Union
from .layers import CorrectedNLoss, CorrectedTLoss
from .utils.datasets import GARCHDataset, FIGARCHDataset, GJRGARCHDataset, LSTMGARCHDataset
from .utils.helpers import generate_ground_data, save_model_params, fit_figarch_parameters, compute_omega, compute_lambda_sequence

__all__: List[str] = [
    'GARCHModel',
    'FIGARCHModel',
    'GJRGARCHModel',
    'GARCHLSTM',
    'CorrectedNLoss',
    'CorrectedTLoss',
    'GARCHDataset',
    'FIGARCHDataset',
    'GJRGARCHDataset',
    'LSTMGARCHDataset',
    'generate_ground_data',
    'save_model_params',
    'fit_figarch_parameters',
    'compute_omega',
    'compute_lambda_sequence'
]

if __package__ or "." in __name__:
    from .models import *
    from .layers import *
    from .utils import *
else:
    # Позволяет запускать файлы напрямую из папки модуля
    from models import *
    from layers import *
    from utils import *