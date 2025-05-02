import numpy as np 
from scipy.optimize import least_squares, fsolve
import json
import os 
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple


def compute_lambda_sequence(d, phi1, beta1, truncation_size):
    # Инициализация списков для хранения delta и lambda
    delta = [1.0]  # delta_{d,0} = 1
    lambdas = []
    
    # Вычисление lambda_1
    lambda1 = phi1 - beta1 + d
    lambdas.append(lambda1)
    
    # Вычисление delta_{d,1} (если требуется)
    if truncation_size >= 2:
        delta1 = (1 - d) / 1  # delta_{d,1} = (1 - d)/1
        delta.append(delta1)
    
    # Вычисление lambda_2
    if truncation_size >= 2:
        lambda2 = (d - beta1) * (beta1 - phi1) + d * (1 - d) / 2
        lambdas.append(lambda2)
    
    # Вычисление delta_{d,k} и lambda_k для k >= 3
    for k in range(3, truncation_size + 1):
        # Вычисление delta_{d,k-1} по рекуррентной формуле
        delta_k_minus_1 = delta[-1] * (k - 2 - d) / (k - 1)
        delta.append(delta_k_minus_1)
        
        # Вычисление lambda_k по рекуррентной формуле
        term = ((k - 1 - d) / k - phi1) * delta_k_minus_1
        lambda_k = beta1 * lambdas[-1] + term
        lambdas.append(lambda_k)
    
    return lambdas[:truncation_size]  # Возвращаем только запрошенное количество элементов



def fit_figarch_parameters(target_lambdas: list, 
                          initial_guess: tuple = (0.5, 0.2, 0.3),
                          maxiter: int = 500) -> tuple:
    """
    Подбирает параметры FIGARCH (d, phi, beta) для заданной последовательности lambda,
    учитывая теоретические ограничения модели.
    
    Параметры:
        target_lambdas (list): Целевые значения lambda последовательности
        initial_guess (tuple): Начальное приближение (d, phi, beta)
        maxiter (int): Максимальное число итераций оптимизации
        
    Возвращает:
        tuple: Оптимальные параметры (d, phi, beta)
    """
    def compute_lambdas(params, n):
        d, phi, beta = params
        # Проверка основных ограничений
        if not (0 < d < 1) or not (beta - d <= phi <= (2 - d)/3):
            return np.full(n, np.nan)
        
        lambdas = [phi - beta + d]
        if n >= 2:
            term = (d - beta)*(beta - phi) + d*(1 - d)/2
            if not (d*(phi - (1-d)/2) <= beta*(d - beta + phi)):
                return np.full(n, np.nan)
            lambdas.append(term)
        
        delta = (1 - d)/1
        for k in range(3, n+1):
            delta *= (k - 2 - d)/(k - 1)
            term = ((k - 1 - d)/k - phi) * delta
            lambdas.append(beta * lambdas[-1] + term)
        
        return np.array(lambdas[:n])

    def objective(params):
        computed = compute_lambdas(params, len(target_lambdas))
        if np.any(np.isnan(computed)):
            return 1e12  # Большой штраф за нарушение ограничений
        return np.mean((computed - target_lambdas)**2)

    # Ограничения в формате для scipy.optimize.minimize
    constraints = (
        {'type': 'ineq', 'fun': lambda x: x[0] - 1e-6},       # d > 0
        {'type': 'ineq', 'fun': lambda x: 1 - x[0] - 1e-6},    # d < 1
        {'type': 'ineq', 'fun': lambda x: x[1] - (x[2] - x[0])},  # phi ≥ beta - d
        {'type': 'ineq', 'fun': lambda x: (2 - x[0])/3 - x[1]},   # phi ≤ (2 - d)/3
        {'type': 'ineq', 'fun': lambda x: x[0]*(x[1] - (1 - x[0])/2) - x[2]*(x[0] - x[2] + x[1])}
    )

    # Границы параметров
    bounds = [
        (1e-6, 1-1e-6),   # d ∈ (0,1)
        (None, None),      # phi
        (1e-6, None)       # beta ≥ 0
    ]

    result = minimize(
        objective,
        x0=np.array(initial_guess),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': maxiter}
    )

    if not result.success:
        print(f"Предупреждение: Оптимизация не сошлась. {result.message}")
    
    return tuple(result.x)








def save_model_params(filename, ground_truth, model_params,model = None):
    if model == None:
        raise(NotImplementedError)

    if model == 'GARCH':
        result = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'ground_truth': {
            'omega': float(ground_truth[0]),
            'alpha': float(ground_truth[1]),
            'beta': float(ground_truth[2]),
        },
        'model_params': {
            'omega': float(model_params[0]),
            'alpha': float(model_params[1]),
            'beta': float(model_params[2])
        }

        }
    
        mode = 'a' if os.path.exists(filename) else 'w'
        with open(filename, mode, encoding='utf-8') as f:
            f.write(json.dumps(result, indent=4) + '\n')  

    if model == 'FIGARCH':
