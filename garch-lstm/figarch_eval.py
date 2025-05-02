import numpy as np 
from scipy.optimize import least_squares, fsolve


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


def fit_lambda_parameters(target_lambdas, initial_guess=None, bounds=None):
    """
    Находит параметры d, phi1, beta1, которые наилучшим образом соответствуют заданным lambda_k.
    
    Параметры:
    target_lambdas : list
        Список целевых значений lambda_k (начиная с lambda_1).
    initial_guess : tuple (d, phi1, beta1), optional
        Начальное предположение для параметров.
    bounds : tuple of tuples, optional
        Границы для параметров: ((d_min, phi1_min, beta1_min), (d_max, phi1_max, beta1_max)).
    
    Возвращает:
    dict
        Словарь с найденными параметрами и информацией об оптимизации.
    """
    if initial_guess is None:
        initial_guess = (0.5, 0.5, 0.5)  # Стандартное начальное предположение
    if bounds is None:
        bounds = ((0, -np.inf, 0), (1, np.inf, np.inf))  # d ∈ (0,1), beta1 ≥ 0
    
    def residuals(params):
        d, phi1, beta1 = params
        computed_lambdas = compute_lambda_sequence(d, phi1, beta1, len(target_lambdas))
        return np.array(computed_lambdas) - np.array(target_lambdas)
    
    result = least_squares(
        residuals,
        x0=initial_guess,
        bounds=bounds,
        method='trf'  # Метод, поддерживающий границы
    )
    
    return result.x[0], result.x[1], result.x[2]


def _constraints_penalty(params: np.ndarray) -> float:
    """Штрафная функция за нарушение ограничений"""
    d, phi1, beta1 = params
    penalty = 0.0
    
    # Основные ограничения
    if not (0 < d < 1):
        penalty += 1e6 * (min(abs(d), abs(d - 1)) + 1)
    
    if not (beta1 - d <= phi1 <= (2 - d)/3):
        penalty += 1e6 * (min(abs(phi1 - (beta1 - d)), abs(phi1 - (2 - d)/3)) + 1)
    
    if not (d*(phi1 - (1 - d)/2) <= beta1*(d - beta1 + phi1)):
        penalty += 1e6
    
    return penalty

def fit_lambda_parameters(target_lambdas: List[float],
                         initial_guess: Tuple[float, float, float] = None,
                         bounds: Tuple[Tuple[float, float, float]] = None,
                         method: str = 'SLSQP') -> Tuple[float, float, float]:
    """
    Подбор параметров FIGARCH с учетом всех ограничений
    
    Parameters:
    -----------
    target_lambdas : List[float]
        Целевые значения lambda последовательности
    initial_guess : Tuple[float, float, float], optional
        Начальное приближение (d, phi1, beta1)
    bounds : Tuple[Tuple[float, float, float]], optional
        Границы параметров
    method : str
        Метод оптимизации (рекомендуется 'SLSQP' или 'trust-constr')
    
    Returns:
    --------
    Tuple[float, float, float]
        Оптимальные параметры (d, phi1, beta1)
    """
    # Стандартные начальные значения
    if initial_guess is None:
        initial_guess = (0.5, 0.2, 0.3)
    
    # Стандартные границы
    if bounds is None:
        bounds = ((1e-6, 1-1e-6),  # d ∈ (0,1)
                 (-np.inf, np.inf), # phi1
                 (1e-6, np.inf))    # beta1 ≥ 0
    
    # Целевая функция с штрафами
    def objective(params):
        d, phi1, beta1 = params
        computed = compute_lambda_sequence(d, phi1, beta1, len(target_lambdas))
        
        # Если нарушены ограничения, возвращаем большую ошибку
        if any(np.isnan(computed)):
            return 1e12 + _constraints_penalty(params)
        
        # MSE + штраф за ограничения
        mse = np.mean((np.array(computed) - np.array(target_lambdas))**2)
        return mse + _constraints_penalty(params)
    
    # Нелинейные ограничения
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[1] - (x[2] - x[0])},  # phi1 ≥ beta1 - d
        {'type': 'ineq', 'fun': lambda x: (2 - x[0])/3 - x[1]},    # phi1 ≤ (2 - d)/3
        {'type': 'ineq', 'fun': lambda x: x[0]*(x[1] - (1 - x[0])/2) - x[2]*(x[0] - x[2] + x[1]))}
    ]
    
    # Оптимизация
    result = minimize(
        objective,
        x0=np.array(initial_guess),
        bounds=bounds,
        method=method,
        constraints=constraints if method in ['SLSQP', 'trust-constr'] else None,
        options={'maxiter': 1000}
    )
    
    return tuple(result.x)

# Пример использования
target_lambdas = [0.6, 0.3, 0.2, 0.15, 0.1]
params = fit_lambda_parameters(target_lambdas)
print(f"Оптимальные параметры: d={params[0]:.4f}, phi1={params[1]:.4f}, beta1={params[2]:.4f}")