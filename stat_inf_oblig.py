import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams, colors, cm, colormaps
from scipy import stats

from matplotlib import rcParams
# Some formatting for nicer plots
params = {
    "text.usetex": True,
    "font.family": "CMU serif",
    "font.serif": ["Computer Modern Serif"],
    "font.size": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,  # a little smaller
    "ytick.labelsize": 9,
    "lines.linewidth": 1.2
}
rcParams.update(params)
plt.rc("figure", titlesize=9)
plt.rc("text.latex", preamble=r"\usepackage{siunitx} \usepackage[T1]{fontenc} \usepackage{xcolor}")


def f_2(x):
    """Density for the sum of two independent uniform random variables on the unit interval."""
    if 0 < x <= 1:
        return x
    elif 1 < x < 2:
        return 2 - x
    return 0


def f_3(x):
    """Density for the sum of three independent uniform random variables on the unit interval."""
    if 0 < x <= 1:
        return 1/2 * x**2
    elif 1 < x <= 2:
        return 1/2 * (6*x - 2*x**2 - 3)
    elif 2 < x < 3:
        return 1/2 * (3 - x)**2
    return 0


def plot_f_1_and_f_2(x_min, x_max):
    xs = np.linspace(x_min, x_max, 1000)
    f_1s = np.array([f_2(x) for x in xs])
    f_2s = np.array([f_3(x) for x in xs])
    fig, ax = plt.subplots(figsize=(3.7, 2.4))
    max_val = max(max(f_1s), max(f_2s))
    pad_y = max_val * 0.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_val + pad_y)
    ax.plot(xs, f_1s, color="blue", label=r"Density $f_2$ of $U_1 + U_2$")
    ax.plot(xs, f_2s, color="orange", label=r"Density $f_3$ of $U_1 + U_2 + U_3$")
    ax.set_xlabel(r"$x$")
    plt.grid()
    plt.tight_layout()
    plt.legend(handlelength=1.2, borderpad=0.3)
    # plt.savefig("densities_of_uniform_sums.svg")
    plt.show()


# plot_f_1_and_f_2(-0.0, 3.0)


def moment_generator_Z_n(t, n):
    return (2 * np.sqrt(n) / t)**n * np.sinh(t / (2 * np.sqrt(n)))**n


def plot_moment_generator():
    ts = np.linspace(-1, 1, 999)    # Avoid t = 0
    ns = [1, 5, 10]
    linestyles = ["solid", "dashed", "dotted"]
    colornorm = colors.Normalize(vmin=min(ns), vmax=max(ns))
    cmap = colormaps["viridis"]
    num_to_col = cm.ScalarMappable(colornorm, cmap)
    fig, ax = plt.subplots(figsize=(3.7, 2.4))
    for i, n in enumerate(ns):
        ax.plot(ts, moment_generator_Z_n(ts, n), label=r"$n={}$".format(n),
                color=num_to_col.to_rgba(n), linestyle=linestyles[i])
    ax.set_xlim(min(ts), max(ts))
    ax.set_ylim(1)
    ax.set_xlabel(r"$t$")
    plt.grid()
    plt.tight_layout()
    plt.legend(handlelength=1.2, borderpad=0.3)
    plt.savefig("moment_gen_funcs_Z_n.svg")
    plt.show()

# plot_moment_generator()

"""Problem 1 e)"""

def estimate_e(n):
    """We estimate e according to the problem set. Let n be the number of realisations of N."""
    Ns = np.zeros(n, dtype=np.uint)
    for i in range(n):
        u_i_sum = 0.0
        N = 0
        while u_i_sum < 1:
            u_i_sum += np.random.uniform(low=0.0, high=1.0)
            N += 1
        Ns[i] = N
        # print(u_is)
    # print(Ns)
    est_e = np.average(Ns)
    est_var_square = np.var(Ns) * n / (n - 1)
    return est_e, est_var_square


def confidence_interval(mu_est, sigma_square_est, n_df):
    """The random variable sqrt(n)(N_avg - e)/sigma_hat is student-t distributed with n-1 degrees of freedom.
    We want a 99% conficence interval, i.e. 0.5 % in either end."""
    n = n_df + 1
    lower_limit, upper_limit = stats.t(df=n_df).ppf((0.005, 0.995))
    # print("{}, {}".format(lower_limit, upper_limit))
    sigma_hat = np.sqrt(sigma_square_est)
    return np.array([lower_limit * sigma_hat / np.sqrt(n) + mu_est, upper_limit * sigma_hat / np.sqrt(n) + mu_est])

N = 10**6
mu_est, sigma_square_est = estimate_e(N)
print("Estimation for e: {}, with variance estimate {}".format(mu_est, sigma_square_est))
print("99% confidence interval for e: [{}, {}]".format(*confidence_interval(mu_est, sigma_square_est, N - 1)))
