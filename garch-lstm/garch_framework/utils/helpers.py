import numpy as np 
from scipy.optimize import least_squares, fsolve
import json
import os 
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Literal
from arch import arch_model

def compute_lambda_sequence(d, phi, beta, truncation_size):
    """
    Вычисляет коэффициенты lambda_k для заданных параметров d, phi, beta и размера усечения T.

    Параметры:
    - d: параметр дробного дифференцирования (float)
    - phi: параметр AR (float)
    - beta: параметр MA (float)
    - truncation_size: количество коэффициентов lambda_k (int)

    Возвращает:
    - Массив коэффициентов lambda_k (numpy array)
    """
    # Инициализация массивов для pi_i и psi_j
    pi = np.zeros(truncation_size)
    psi = np.zeros(truncation_size)
    lambda_coeffs = np.zeros(truncation_size)

    # Вычисление коэффициентов pi_i (разложение (1 - B)^d)
    pi[0] = 1.0
    for i in range(1, truncation_size):
        pi[i] = pi[i-1] * (i - 1 - d) / i

    # Вычисление коэффициентов psi_j (разложение (1 - phi B)/(1 - beta B))
    psi[0] = 1.0
    if truncation_size > 1:
        psi[1] = beta - phi
    for j in range(2, truncation_size):
        psi[j] = beta * psi[j-1]

    # Вычисление коэффициентов lambda_k
    for k in range(truncation_size):
        gamma_k = 0.0
        for i in range(k + 1):
            gamma_k += psi[k - i] * pi[i]
        
        if k == 0:
            lambda_coeffs[k] = 1 - gamma_k
        else:
            lambda_coeffs[k] = -gamma_k

    return lambda_coeffs[1:]



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


def compute_omega(weights, eps_squared, vol_series, trunc=None):
    weights = weights.numpy()
    truncation_size = len(weights[:trunc])
    
    pred = np.sum(weights[:truncation_size] * eps_squared[:truncation_size])
    omega = vol_series[truncation_size] - pred
    
    return max(omega, 1e-6)  




def save_model_params(filename, ground_truth, model_params, model: Literal["GARCH", "FIGARCH", "GJR-GARCH"], residuals_squared = None, volatility= None):
    if model == None:
        raise NotImplementedError

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
        return None

    if model == 'FIGARCH':
        if residuals_squared or volatility == None:
            raise NotImplementedError
        computed_params = fit_figarch_parameters(model_params)
        computed_omega = compute_omega(model_params, residuals_squared, volatility)

        result = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'ground_truth': {
            'omega' : float(ground_truth[0]),
            'd': float(ground_truth[1]),
            'phi': float(ground_truth[2]),
            'beta': float(ground_truth[3]),
        },
        'model_params': {
            'omega' : float(computed_omega),
            'd': float(computed_params[0]),
            'phi': float(computed_params[1]),
            'beta': float(computed_params[2])
        }

        }

        mode = 'a' if os.path.exists(filename) else 'w'
        with open(filename, mode, encoding='utf-8') as f:
            f.write(json.dumps(result, indent=4) + '\n')
        return None
    
    if model == "GJRGARCH":
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'ground_truth': {
                'omega': float(ground_truth[0]),
                'alpha': float(ground_truth[1]),
                'gamma' : float(ground_truth[2]),
                'beta': float(ground_truth[3]),
            },
            'model_params': {
                'omega': float(model_params[0]),
                'alpha': float(model_params[1]),
                'gamma' : float(model_params[2]),
                'beta': float(model_params[3])
            }

        }
        
        # Запись в файл (дозапись)
        mode = 'a' if os.path.exists(filename) else 'w'
        with open(filename, mode, encoding='utf-8') as f:
            f.write(json.dumps(result, indent=4) + '\n')  # Добавляем перевод строки
        return None
    
    return None
    
def generate_ground_data(mode:Literal["GARCH", "FIGARCH", "GJRGARCH"], omega= None, alpha=None, beta=None, gamma = None, d= None, phi = None,n=1000):
    if mode == "GARCH":
        if omega == None or alpha == None or beta == None:
            raise NotImplementedError
        am = arch_model(None, mean='Zero', vol='GARCH', p=1, q=1, power = 2.0) #Остатки просто получаются умножением волатильности на кси ~N(0,1)
        params = np.array([omega, alpha, beta])
        am_data = am.simulate(params, n)

        return am_data['data'].to_numpy(), am_data['volatility'].to_numpy()
    
    elif mode == "FIGARCH":
        if omega == None or d == None or phi == None or beta == None:
            raise NotImplementedError

        am = arch_model(None, mean='Zero', vol='FIGARCH', p=1, q=1, power=2.0)
        params = np.array([omega, d, phi, beta])
        am_data = am.simulate(params, n)
        return am_data['data'].to_numpy(), am_data['volatility'].to_numpy()
    
    elif mode == "GJRGARCH":
        if omega == None or alpha == None or beta == None or gamma == None:
            raise NotImplementedError
        
        am = arch_model(None, mean='Zero', vol='GARCH', p=1, o=1, q=1, power = 2) #Остатки просто получаются умножением волатильности на кси ~N(0,1)
        params = np.array([omega, alpha, gamma, beta])
        am_data = am.simulate(params, n)

        return am_data['data'].to_numpy(), am_data['volatility'].to_numpy()


    return None

        
