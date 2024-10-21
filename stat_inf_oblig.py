import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams, colors, cm, colormaps
from matplotlib.cm import ScalarMappable
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
# mu_est, sigma_square_est = estimate_e(N)
# print("Estimation for e: {}, with variance estimate {}".format(mu_est, sigma_square_est))
# print("99% confidence interval for e: [{}, {}]".format(*confidence_interval(mu_est, sigma_square_est, N - 1)))

def sample_random_normal_triplet(n):
    """Sampling n triplets of random variables (X, Y, Z), where X, Y, Z are independent standard normals."""
    return np.random.normal(size=(n, 3))

def spherical_transform(x_y_zs):
    """Takes in an array of shape (n, 3) and transforms each triplet to spherical coordinates (R, A, B).
    A in [0, pi]."""

    def spherical(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return np.array([r, np.arccos(z / r), np.sign(y) * np.arccos(x / np.sqrt(x**2 + y ** 2))])

    r_a_bs = np.zeros_like(x_y_zs)
    for i in range(x_y_zs.shape[0]):
        r_a_bs[i] = spherical(*x_y_zs[i, :])
    return r_a_bs

# rand_samples = sample_random_normal_triplet(10)
# rand_spher = spherical_transform(rand_samples)
# print(rand_samples)
# print(rand_spher)


def plot_histograms(n):
    x_y_zs = sample_random_normal_triplet(n)
    r_a_bs = spherical_transform(x_y_zs)
    # print(x_y_zs)
    # print(r_a_bs)
    fig, axs = plt.subplots(ncols=3, figsize=(5.8, 2.3))
    colornorm = colors.Normalize(vmin=0, vmax=2)
    cmap = colormaps["viridis"]
    num_to_col = cm.ScalarMappable(norm=colornorm, cmap=cmap)
    linspace1 = np.linspace(0, max(r_a_bs[:, 0]), 1000)
    linspace2 = np.linspace(0, np.pi, 1000)
    linspace3 = np.linspace(-np.pi, np.pi, 100)
    rs = np.sqrt(2 / np.pi) * linspace1 ** 2 * np.exp(-linspace1**2 / 2)
    a_arr = np.sin(linspace2) / 2
    b_arr = np.array([1 / (2 * np.pi) for i in linspace3])
    linspaces = [linspace1, linspace2, linspace3]
    densities = [rs, a_arr, b_arr]
    for i, (ax, linspace, density) in enumerate(zip(axs, linspaces, densities)):
        ax.hist(r_a_bs[:, i], bins='auto', color=num_to_col.to_rgba(i), density=True)
        ax.plot(linspace, density, color="black", label=r"$g_{}$".format(i+1))
        ax.grid()
        ax.legend()
    axs[0].set_xlim(0, max(r_a_bs[:, 0]))
    axs[0].set_title(r"$R$")
    axs[0].set_xlabel(r"$r$")
    axs[0].set_xticks(np.array([0, 1, 2, 3, 4]))

    axs[1].set_xlim(0, np.pi)
    axs[1].set_title(r"$A$")
    axs[1].set_xlabel(r"$a$")
    axs[1].set_xticks(np.pi * np.array([0, 1 / 4, 1/2, 3/4, 1]))
    axs[1].set_xticklabels([0, r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])

    axs[2].set_xlim(-np.pi, np.pi)
    axs[2].set_title(r"$B$")
    axs[2].set_xlabel(r"$b$")
    axs[2].set_xticks(np.pi * np.array([-1, -2 / 3, -1/3, 0, 1/3, 2/3, 1]))
    axs[2].set_xticklabels([r"$-\pi$", r"$-\frac{2}{3}\pi$", r"$-\frac{1}{3}\pi$", 0 , r"$\frac{1}{3}\pi$", r"$\frac{2}{3}\pi$", r"$\pi$"])
    plt.tight_layout()
    plt.savefig(fname="spherically_transformed_triplett.svg")
    plt.show()

plot_histograms(1000)
