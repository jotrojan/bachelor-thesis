from utils.methods import rk4, trapezoid, secant
import numpy as np
from utils.utilities import (
    y_vacuum,
    get_ODE_function,
    get_secant_function,
    psi_is_positive,
    round_to_tolerance,
    unique,
)


def bound_critical_parameter(
    _get_rho,
    rk_start,
    rk_end,
    rk_step,
    secant_tolerance,
    secant_max_iters,
    secant_initial_step,
    parameter_max,
    alpha_max,
    alpha_min,
    max_iters=100,
    minimal_seeked_bound_distance=1e-1,
    verbose=True,
    scale=2
):
    alpha_upper_bound = alpha_max
    alpha_lower_bound = alpha_min

    results = ((None, None), (None, None), False)

    parameter_start = None
    parameter_end = None
    if parameter_max is None:  
        r_values = np.arange(rk_start, rk_end + rk_step, rk_step)
        unit_density = _get_rho(1)
        unit_density_values = [unit_density(r) for r in r_values]
        integrand = [rho * r**2 for r, rho in zip(r_values, unit_density_values)]
        integral_result = trapezoid(r_values, integrand)

        if integral_result == 0:
            amplitude_bound = 1.0 # ad hoc safe default
        else:
            amplitude_bound = (1 / (2 * np.pi)) * (4**4 / 5**5) * rk_start / integral_result
        if verbose:
            print(f"Upper bound for amplitude is approximately: {amplitude_bound} \n")

        parameter_max = amplitude_bound

    current_parameter = parameter_max
    tmp = parameter_max  # tmp tracks the last known parameter, for which both guesses failed

    for iteration in range(max_iters):
        _rho = _get_rho(current_parameter)
        _sec = get_secant_function(_rho, rk_start, rk_end, rk_step)

        iterations_top, no_iters_top, success_top = secant(
            _sec, secant_tolerance, secant_max_iters, alpha_upper_bound, secant_initial_step,
        )
        iterations_bot, no_iters_bot, success_bot = secant(
            _sec, secant_tolerance, secant_max_iters, alpha_lower_bound, secant_initial_step,
        )

        if not success_bot or not success_top:
            if not success_bot and not success_top:
                tmp = current_parameter

            if verbose:
                print(f"[{iteration}] {current_parameter:.2e} at least one secant did not converge.")

            current_parameter /= scale
            continue

        _ode = get_ODE_function(_rho)
        root_bot = iterations_bot[-1]
        root_top = iterations_top[-1]

        positive_top = psi_is_positive(
            _ode, rk_start, y_vacuum(root_top, rk_start), rk_end, rk_step,
        )
        positive_bot = psi_is_positive(
            _ode, rk_start, y_vacuum(root_bot, rk_start), rk_end, rk_step,
        )


        if (positive_top and (root_top > 0)) and (positive_bot and (root_bot > 0)):

            # check if the distance between roots is large enough (it is not the same root)
            if np.abs(root_top - root_bot) > minimal_seeked_bound_distance:
                if verbose:
                    print(f"[{iteration}] {current_parameter:.2e} converged! \n")
                    print(f" top: {round_to_tolerance(root_top, secant_tolerance)} after {no_iters_top} secant iterations")
                    print(f" bot: {round_to_tolerance(root_bot, secant_tolerance)} after {no_iters_bot} secant iterations")

                results = ((current_parameter, tmp), (root_bot, root_top), True)
                break

        if verbose:
            print(f"[{iteration}] {current_parameter:.2e} at least one secant guess converged to a non-valid root.")

        current_parameter /= scale

    # summary of results

    if results[2]:
        alpha_start = results[1][0]
        alpha_end = results[1][1]
        parameter_start = results[0][0]
        parameter_end = results[0][1]

        print("\nFound bounds for critical parameter and alpha values:")
        print(f"Alpha bounds: {alpha_start} {alpha_end}")
        print(f"Parameter bounds: {parameter_start} {parameter_end}")

    elif parameter_start is not None and parameter_end is not None and parameter_end == parameter_start:
        print("Converged too fast.")
        parameter_end = parameter_max
    else:
        print("Did not work.")

    return results


def locate_critical_parameter(
    _get_rho,
    start,
    end,
    alpha_start,  
    alpha_end,           
    rk_start,
    rk_end,
    rk_step,
    secant_tolerance,
    secant_max_iters,
    secant_initial_step,
    bisection_tolerance,
    bisection_max_iters,
    bisection_root_tol,
):


    iteration = 0
    iteration_steps = []

    a = start
    b = end


    _alphas = [] 
    _param = None 


    while (b - a) > bisection_tolerance and iteration < bisection_max_iters:
        midpoint = (a + b) / 2

        print(f"[{iteration}] ({a:.2e}  | {midpoint:.2e} | {b:.2e})", end="")

        rho = _get_rho(midpoint)
        ode = get_ODE_function(rho)
        sec = get_secant_function(rho, rk_start, rk_end, rk_step)

        roots = []


        iterations_1, _, success_1 = secant(
            sec,
            secant_tolerance,
            secant_max_iters,
            alpha_start,
            secant_initial_step
        )
        if success_1:
            final_alpha_1 = iterations_1[-1]
            is_positive_1 = psi_is_positive(
                ode,
                rk_start,
                y_vacuum(final_alpha_1, rk_start),
                rk_end,
                rk_step
            )
            if is_positive_1 and final_alpha_1 > 0:
                roots.append(final_alpha_1)
                print(" ↑", end="")

        iterations_2, _, success_2 = secant(
            sec,
            secant_tolerance,
            secant_max_iters,
            alpha_end,
            secant_initial_step
        )
        if success_2:
            final_alpha_2 = iterations_2[-1]
            is_positive_2 = psi_is_positive(
                ode,
                rk_start,
                y_vacuum(final_alpha_2, rk_start),
                rk_end,
                rk_step
            )
            if is_positive_2 and final_alpha_2 > 0:
                roots.append(final_alpha_2)
                print(" ↓", end="")

        number_of_roots = len(roots)

        if number_of_roots == 0:
            b = midpoint
            print(f" <<< {roots}")
        else:
            _alphas = roots
            _param = midpoint
            print(f" >>> {roots}")
            a = midpoint

        if number_of_roots == 2:
            alpha_first = roots[0]
            alpha_second = roots[1]

            if abs(alpha_first - alpha_second) < bisection_root_tol:
                break

        iteration_steps.append((a, b, midpoint, number_of_roots, roots))

        iteration += 1

    if iteration == bisection_max_iters:
        print("Failed. Maximum number of iterations reached.")

    print("\n Done. \n")


    if _alphas:
        alpha_critical = np.mean(_alphas)
        alpha_std = np.std(_alphas)
        param_critical = _param
        print(f"critical alpha: {alpha_critical} (±{alpha_std})")
        print(f"critical parameter: {param_critical} (±{bisection_tolerance})")
    else:
        alpha_critical = None
        param_critical = None
        print("No valid alpha roots discovered in the bisection range.")

    return alpha_critical, param_critical, iteration_steps

def rest_mass(
    r: float,
    r_prime: np.ndarray,
    psi_at_r_prime: np.ndarray,
    rho_at_r_prime: np.ndarray,
    exponent: int = 6,
) -> float:
    mask = r_prime <= r
    return (
        4 * np.pi * trapezoid(
            rho_at_r_prime[mask]
            * psi_at_r_prime[mask] ** exponent
            * r_prime[mask] ** 2,
            r_prime[mask],
        )
    )


def bruteforce_parameter_and_energy_relationship(get_rho, parameter_search_range, alpha_search_range, rk_params, secant_params, exponent = 5, unique_root_tolerance = 1e-4):
    ADM_mass_vals = []
    rest_mass_vals = []
    parameters = []
    horizons = []

    for j, parameter in enumerate(parameter_search_range):
        print(f"[{j+1}/{len(parameter_search_range)}]", end="")

        sec_tol, sec_iters, sec_step = secant_params
        rk_start, rk_end, rk_step = rk_params

        rho = get_rho(parameter)
        ode = get_ODE_function(rho, exponent=exponent)
        sec = get_secant_function(rho, *rk_params, exponent=exponent)


        valid_roots = []

        for i, alpha in enumerate(alpha_search_range):
            res, _, success = secant(sec, sec_tol, sec_iters, alpha, sec_step)

            if not success or len(res) == 0:
                continue

            possible_root = res[-1]

            if possible_root < 0:
                continue

            if  psi_is_positive(ode, rk_start, y_vacuum(possible_root, rk_start), rk_end, rk_step):
                valid_roots.append(possible_root)

        results = unique(valid_roots, tol=unique_root_tolerance)

        if len(results) != 0:
            print(" ✓ ", end="")
        else:
            print(" ✗ ", end="")

        print(f"parameter = {parameter}, (unique tolerance {unique_root_tolerance}), ", end="")
        print(f"{len(alpha_search_range)-len(valid_roots)} out of {len(alpha_search_range)} attempts failed, found {len(results)} solutions: ") 

        for i, result in enumerate(results):
            print(f"  [{i+1}] {round_to_tolerance(result[0], unique_root_tolerance)} (err {result[1]/result[0]*100:.2g} %, {result[2]}/{len(valid_roots)}, ", end="")
            x,y = rk4(ode, rk_start, y_vacuum(result[0], rk_start), rk_end, rk_step)
            print(f"ψ'(0) ≈ {y[:,1][-1]:.1g})", end="")

            r_vals = np.flip(x)
            psi_vals = np.flip(y[:, 0])
            rho_vals = np.array([rho(r) for r in r_vals])
 
            _rest_mass = rest_mass(r_vals[-1], r_vals, psi_vals, rho_vals, exponent=exponent+1)
            _ADM_mass = 2 * result[0]
            print(f" rest mass = {_rest_mass:.2g}, ADM energy = {_ADM_mass:.2g}")
            ADM_mass_vals.append(_ADM_mass)
            rest_mass_vals.append(_rest_mass)
            parameters.append(parameter)
        
            r = x
            psi = y[:,0]
            dpsi = y[:,1]

            _f = [dpsi[i] + 0.5 * psi[i] / r[i] for i in range(len(r))]

            for i in range(1, len(r)):
                if np.sign(_f[i]) != np.sign(_f[i-1]):
                    print(f"   [!] found apparent horizon at roughly around r = {r[i]}")
                    horizons.append([parameter, _ADM_mass, _rest_mass, r[i]])
                    break

        print()

    return ADM_mass_vals, rest_mass_vals, parameters, horizons

