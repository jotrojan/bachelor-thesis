import numpy as np
from numba import njit
from typing import Callable
from utils.methods import rk4


@njit
def y_vacuum(a, r) -> np.ndarray:

    y1 = 1.0 + a / r
    y2 = - a / (r**2)

    return np.array([y1, y2])

@njit
def psi_is_positive(
    func: Callable[[float, np.ndarray], np.ndarray],
    x0: float,
    y0: np.ndarray,
    xn: float,
    h: float,
) -> bool:
    y = y0
    for x in np.arange(x0, xn + h, h):
        k1 = h * func(x, y)
        k2 = h * func(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * func(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * func(x + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if y[0] < 0:  # integrated psi has a negative function value
            return False
    return True


def get_ODE_function(
    rho: Callable[[float], float],
    exponent: int = 5,
) -> Callable[[float, np.ndarray], np.ndarray]:

    @njit
    def _ode_function(x: float, y: np.ndarray) -> np.ndarray:
        y1 = y[1]
        y2 = -2 / x * y[1] - 2 * np.pi * rho(x) * y[0] ** exponent
        return np.array([y1, y2])

    return _ode_function

def get_secant_function(
    rho: Callable[[float], float],
    rk_start: float,
    rk_end: float,
    rk_step: float,
    exponent: int = 5,
):

    _ode = get_ODE_function(rho, exponent=exponent)

    def _secant_function(alpha: float) -> float:
        _, y = rk4(_ode, rk_start, y_vacuum(alpha, rk_start), rk_end, rk_step)
        return float(y[-1][1])

    return _secant_function



def round_to_tolerance(value, tolerance):
    return round(value, -int(np.log10(tolerance)))


def geospace(start, end, steps=50, ratio=0.9):
    k = np.arange(steps)
    x = start + (end - start) * (1 - ratio**k) / (1 - ratio ** (steps - 1))
    return x


def unique(values, tol=1e-3):
    arr = np.sort(np.array(values, dtype=float))
    
    if arr.size == 0:
        return []
    
    diffs = np.diff(arr)
    
    boundaries = np.concatenate(([0], np.where(diffs > tol)[0] + 1, [len(arr)]))

    results = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block = arr[start:end]
        block_mean = block.mean()
        block_std = block.std()
        results.append([block_mean, block_std, block.size])

    return results


def get_monotonic_segments(values, other_values=None):
    
    interval_partition_indices = []
    interval_partition_indices.append(0)

    increasing = True
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            if increasing:
                interval_partition_indices.append(i - 1)

                increasing = False
            else:
                continue
        else:
            if not increasing:
                interval_partition_indices.append(i - 1)
                increasing = True
            else:
                continue
    interval_partition_indices.append(len(values) - 1)

    subintervals = []
    
    # separate values into subarrays based on the indices
    for i in range(len(interval_partition_indices) - 1):
        start = interval_partition_indices[i]
        end = interval_partition_indices[i + 1] + 1
        subintervals.append(values[start:end])

    if other_values is not None:
        other_subintervals = []
        for i in range(len(interval_partition_indices) - 1):
            start = interval_partition_indices[i]
            end = interval_partition_indices[i + 1] + 1
            other_subintervals.append(other_values[start:end])
        return subintervals, other_subintervals
    else:
        # if no other values are provided, return only the subintervals
        return subintervals