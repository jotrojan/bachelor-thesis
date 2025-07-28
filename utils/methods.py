from typing import Tuple, Callable
import numpy as np
from numba import njit

@njit
def rk4(
    func: Callable[[float, np.ndarray], np.ndarray],
    x0: float,
    y0: np.ndarray,
    xn: float,
    h: float,
) -> Tuple[np.ndarray, np.ndarray]:

    x: np.ndarray = np.arange(x0, xn + h, h)
    n: int = len(x)
    y: np.ndarray = np.zeros((n, len(y0)))

    y[0] = y0

    for i in range(n - 1):

        k1 = h * func(x[i], y[i])
        k2 = h * func(x[i] + 0.5 * h, y[i] + 0.5 * k1)
        k3 = h * func(x[i] + 0.5 * h, y[i] + 0.5 * k2)
        k4 = h * func(x[i] + h, y[i] + k3)

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    return x, y


def secant(
    func: Callable[[float], float],
    tol: float,
    max_iters: int,
    init_guess: float,
    init_step: float,
) -> Tuple[np.ndarray, int, bool]:

    machine_precision = np.finfo(float).eps
    x0 = init_guess
    x1 = init_guess + init_step
    fx0 = func(x0)
    fx1 = func(x1)
 
    for i in range(max_iters - 1):

        if np.abs(x1 - x0) < machine_precision:
            if np.abs(fx1) < tol:
                return np.array([x1]), i+1, True
            else:
                return np.array([x1]), i+1, False
        
        denominator = fx1 - fx0

        if np.abs(denominator) < machine_precision:
            if np.abs(fx1) < tol:
                return np.array([x1]), i+1, True
            else:
                return np.array([x1]), i+1, False
            
        x_new = x1 - fx1 * (x1 - x0) / denominator
        f_new = func(x_new)
    
        if np.abs(f_new) < tol:
            return np.array([x_new]), i+1, True
        
        x0 = x1
        x1 = x_new

        fx0 = fx1
        fx1 = f_new
        
    return np.array([x1]), max_iters, False 

def trapezoid(f_x: np.ndarray, x: np.ndarray) -> float:
    N = len(x)
    integral = 0
    for k in range(1, N):
        delta_x_k = x[k] - x[k - 1]
        integral += 0.5 * (f_x[k] + f_x[k - 1]) * delta_x_k
    return integral


def false_position(
    func: Callable[[float], float],
    tol: float,
    max_iters: int,
    a: float,
    b: float,
) -> Tuple[float, int, bool]:

    fa = func(a)
    fb = func(b)

    if fa * fb > 0:
        return np.nan, 0, False

    for i in range(max_iters):
        c = (a * fb - b * fa) / (fb - fa)
        fc = func(c)

        if np.abs(fc) < tol:
            return c, i, True

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return np.nan, max_iters, False