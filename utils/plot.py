from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# for LaTeX compatibility

textwidth = 15.99773 # cm
textheight = 22.69678 # cm
cm_to_inch = 0.3937007874
golden_ratio = (np.sqrt(5) - 1) / 2

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}') 

def formatter(x, pos):
        if np.isclose(x, 0):
            return "0"
        scientific_notation = f"{x:.2e}"
        base, exponent = scientific_notation.split("e")
        return r"${:.2g} \cdot 10^{{{}}}$".format(float(base), int(exponent))

def blend_color(color, alpha, background='white'):
    fg = mcolors.to_rgba(color, alpha)
    bg = mcolors.to_rgba(background)
    return [fg[i] * alpha + bg[i] * (1 - alpha) for i in range(3)]

def draw_horizons(horizons, exponent, delta=0.6, arrow_length=0.1):

    horizon_parameters = [horizon_info[0] for horizon_info in horizons]

    if len(horizon_parameters) == 0:
        print("no horizons found")
    else:

        if exponent <= 0:
            max_parameter = min(horizon_parameters)
        else:
            max_parameter = max(horizon_parameters)


        max_parameter_critical_index = horizon_parameters.index(max_parameter)
        max_parameter_E_ADM = horizons[max_parameter_critical_index][1]
        max_parameter_rest_mass = horizons[max_parameter_critical_index][2]

        print(
            f"first horizon found at approx. parameter {max_parameter:.2g} with ADM energy {max_parameter_E_ADM:.2g} and rest mass {max_parameter_rest_mass:.2g}"
        )

        # hacky way to draw arrows
        # cosmetic, works only if ADM_E and rest_mass are different
        plt.annotate(
            "",
            xy=(max_parameter, (1 - delta) * max_parameter_E_ADM),
            xytext=(max_parameter, (1 + delta) * max_parameter_rest_mass),
            arrowprops=dict(arrowstyle="-", lw=1, ls="-", color="black"),
        )

        if exponent <= 0:
            plt.annotate(
            "",
            xy=(
                max_parameter, 
                (1 - 0.5 * delta) * max_parameter_E_ADM),
            xytext=(
                max_parameter * (1 + 0.5 * arrow_length),
                (1 - 0.5 * delta) * max_parameter_E_ADM,
            ),
            arrowprops=dict(arrowstyle="<-", lw=1, ls="-", color="black"),
            )
        else:
            plt.annotate(
            "",
            xy=(
                max_parameter, 
                (1 - 0.5 * delta) * max_parameter_E_ADM),
            xytext=(
                max_parameter * (1 - 0.5 * arrow_length),
                (1 - 0.5 * delta) * max_parameter_E_ADM,
            ),
            arrowprops=dict(arrowstyle="<-", lw=1, ls="-", color="black"),
            )


        plt.text(
            max_parameter,
            (1 - delta) * max_parameter_E_ADM,
            "app. horizons",
            ha="center",
            va="top",
            fontsize="small",
        )