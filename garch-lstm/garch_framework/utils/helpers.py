import numpy as np 
from scipy.optimize import least_squares, fsolve
import json
import os 
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Literal
from arch import arch_model


def compute_lambda_sequence(d: float, phi: float, beta: float, truncation_size: int) -> np.ndarray:
    """
    Вычисляет коэффициенты lambda_k для модели FIGARCH.

    Параметры:
    - d: параметр дробного дифференцирования (должен быть в диапазоне [0, 1])
    - phi: параметр AR (должен быть в диапазоне (-1, 1))
    - beta: параметр MA (должен быть в диапазоне (-1, 1))
    - truncation_size: количество коэффициентов lambda_k (должно быть >= 1)

    Возвращает:
    - Массив коэффициентов lambda_k (размером truncation_size-1, так как lambda_0 исключается)
    """
    # Инициализация массивов
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

    return lambda_coeffs[1:]  # Исключаем lambda_0


def fit_figarch_parameters(lambda_target: np.ndarray, 
                          truncation_size: int, 
                          initial_guess: Tuple[float, float, float] = (0.5, 0.2, 0.3), 
                          method: str = 'L-BFGS-B') -> Tuple[Tuple[float, float, float], float]:
    """
    Оценивает параметры FIGARCH (d, phi, beta) по заданным коэффициентам lambda_k.

    Параметры:
    - lambda_target: целевые значения lambda_k (размер должен быть truncation_size-1)
    - truncation_size: полный размер усечения (включая lambda_0)
    - initial_guess: начальные значения (d, phi, beta)
    - method: метод оптимизации (поддерживаются методы scipy.optimize.minimize)

    Возвращает:
    - Кортеж с оцененными параметрами (d, phi, beta)
    - Значение функции потерь на оптимуме
    """
    truncation_size = truncation_size + 1  # Учитываем lambda_0

    def loss_function(params):
        d, phi, beta = params
        # Ограничения на параметры
        if not (0 <= d <= 1 and abs(phi) < 1 and abs(beta) < 1):
            return np.inf
        model_lambda = compute_lambda_sequence(d, phi, beta, truncation_size)
        return np.sum((model_lambda - lambda_target)**2)

    result = minimize(loss_function, initial_guess, method=method)

    return tuple(result.x), result.fun


def compute_omega(weights: np.ndarray, 
                 eps_squared: np.ndarray, 
                 vol_series: np.ndarray, 
                 trunc: int = None) -> float:
    """
    Вычисляет параметр omega для модели FIGARCH.

    Параметры:
    - weights: весовые коэффициенты lambda_k
    - eps_squared: квадраты остатков модели
    - vol_series: ряд волатильностей
    - trunc: размер усечения (если None, используется весь массив weights)

    Возвращает:
    - Оцененное значение omega (не меньше 1e-6 для устойчивости)
    """
    truncation_size = len(weights[:trunc])
    
    pred = np.sum(np.flip(weights[:truncation_size]) * eps_squared[:truncation_size])
    omega = vol_series[truncation_size+1] - pred
    
    return max(omega, 1e-6)  # Гарантируем положительность


def save_model_params(filename: str, 
                    ground_truth: Tuple, 
                    model_params: Tuple, 
                    model: Literal["GARCH", "FIGARCH", "GJR-GARCH"], 
                    residuals_squared: np.ndarray = None, 
                    volatility: np.ndarray = None) -> None:
    """
    Сохраняет параметры модели в JSON файл.

    Параметры:
    - filename: путь к файлу для сохранения
    - ground_truth: истинные параметры модели
    - model_params: оцененные параметры модели
    - model: тип модели ("GARCH", "FIGARCH" или "GJR-GARCH")
    - residuals_squared: квадраты остатков (требуется для FIGARCH)
    - volatility: ряд волатильностей (требуется для FIGARCH)
    """
    if model is None:
        raise NotImplementedError("Не указан тип модели")

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
    
    elif model == 'FIGARCH':
        if residuals_squared is None or volatility is None:
            raise ValueError("Для FIGARCH требуются residuals_squared и volatility")
            
        computed_params = fit_figarch_parameters(model_params, truncation_size=len(model_params))
        computed_omega = compute_omega(model_params, residuals_squared, volatility)

        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'ground_truth': {
                'omega': float(ground_truth[0]),
                'd': float(ground_truth[1]),
                'phi': float(ground_truth[2]),
                'beta': float(ground_truth[3]),
            },
            'model_params': {
                'omega': float(computed_omega),
                'd': float(computed_params[0][0]),
                'phi': float(computed_params[0][1]),
                'beta': float(computed_params[0][2])
            }
        }
    
    elif model == "GJRGARCH":
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'ground_truth': {
                'omega': float(ground_truth[0]),
                'alpha': float(ground_truth[1]),
                'gamma': float(ground_truth[2]),
                'beta': float(ground_truth[3]),
            },
            'model_params': {
                'omega': float(model_params[0]),
                'alpha': float(model_params[1]),
                'gamma': float(model_params[2]),
                'beta': float(model_params[3])
            }
        }
    
    else:
        raise NotImplementedError(f"Модель {model} не поддерживается")

    # Запись в файл
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode, encoding='utf-8') as f:
        f.write(json.dumps(result, indent=4) + '\n')


def generate_ground_data(mode: Literal["GARCH", "FIGARCH", "GJRGARCH"],
                       omega: float = None,
                       alpha: float = None,
                       beta: float = None,
                       gamma: float = None,
                       d: float = None,
                       phi: float = None,
                       n: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерирует синтетические данные для заданной модели волатильности.

    Параметры:
    - mode: тип модели ("GARCH", "FIGARCH" или "GJRGARCH")
    - omega: параметр постоянной волатильности (должен быть > 0)
    - alpha: параметр ARCH (должен быть >= 0)
    - beta: параметр GARCH (должен быть >= 0)
    - gamma: параметр асимметрии для GJR-GARCH
    - d: параметр дробного интегрирования для FIGARCH (0 <= d <= 1)
    - phi: параметр AR для FIGARCH
    - n: количество наблюдений для генерации

    Возвращает:
    - Кортеж (остатки, волатильности)
    """
    if mode == "GARCH":
        if omega is None or alpha is None or beta is None:
            raise ValueError("Для GARCH требуются omega, alpha и beta")
        am = arch_model(None, mean='Zero', vol='GARCH', p=1, q=1, power=2.0)
        params = np.array([omega, alpha, beta])
        am_data = am.simulate(params, n)
        return am_data['data'].to_numpy(), am_data['volatility'].to_numpy()
    
    elif mode == "FIGARCH":
        if omega is None or d is None or phi is None or beta is None:
            raise ValueError("Для FIGARCH требуются omega, d, phi и beta")
        am = arch_model(None, mean='Zero', vol='FIGARCH', p=1, q=1, power=2.0)
        params = np.array([omega, d, phi, beta])
        am_data = am.simulate(params, n)
        return am_data['data'].to_numpy(), am_data['volatility'].to_numpy()
    
    elif mode == "GJRGARCH":
        if omega is None or alpha is None or beta is None or gamma is None:
            raise ValueError("Для GJR-GARCH требуются omega, alpha, gamma и beta")
        am = arch_model(None, mean='Zero', vol='GARCH', p=1, o=1, q=1, power=2)
        params = np.array([omega, alpha, gamma, beta])
        am_data = am.simulate(params, n)
        return am_data['data'].to_numpy(), am_data['volatility'].to_numpy()

    raise ValueError(f"Неизвестный тип модели: {mode}")