import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams, colors, cm, colormaps
from scipy import stats

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



def read_csv(fname):
    """Manual reading into list. Handling strings before reading into np.array."""
    list = []
    with open(fname, "r") as f:
        for fline in f:
            line = []
            for entry in fline.strip().split(","):
                try:
                    line.append(float(entry))
                except ValueError:
                    if entry == "NA":
                        line.append(np.NAN)
                    else:
                        line.append(entry)
            list.append(line)

    names = [list[i][0] for i in range(1, len(list))]
    body_brain = np.array([[list[i][-1], list[i][-2]] for i in range(1, len(list))])
    return names, body_brain


def filter_data(names, body_brain):
    """Wish to filter out the animals where the brain-size is not available."""
    names_updated = [name for (name, brain) in zip(names, body_brain[:, 1]) if not np.isnan(brain)]
    body_brain_updated = np.array([pair for pair in body_brain if not np.isnan(pair[1])])
    return names_updated, body_brain_updated


def plot_filtered_data():
    # Fetching data:
    fname = "msleep.csv"
    names, body_brain = filter_data(*read_csv(fname))

    pearson_corr, p_val = stats.pearsonr(np.log(body_brain[:, 0]), np.log(body_brain[:, 1]))

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(4.5, 2.3))
    # Using viridis because I like the colour palette.
    color_norm = colors.Normalize(vmin=-0.5, vmax=1.5)
    color_map = colormaps["viridis"]
    num_to_color = cm.ScalarMappable(norm=color_norm, cmap=color_map)
    ax1.scatter([w for w in body_brain[:, 0] if w < 2000],
                [bw for (w, bw) in zip(body_brain[:, 0], body_brain[:, 1]) if w < 2000],
                color=num_to_color.to_rgba(0), s=10)
    ax2.scatter(body_brain[:, 0], body_brain[:, 1], color=num_to_color.to_rgba(1), s=10)
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax1.set_xlabel("Body weight [kg]")
    ax1.set_ylabel("Brain weight [kg]")
    ax1.grid()

    ax2.set_xlabel("log(body weight)")
    ax2.set_ylabel("log(brain weight)")
    ax2.grid()
    ax2.text(0.02, 0.97, r"$r_{} = {}$".format("{xy}", round(pearson_corr, 3)), verticalalignment="top")
    plt.tight_layout()
    # plt.savefig("body_vs_brain_weight.svg")
    plt.show()


# plot_filtered_data()


def pearson_corr_conf_int():
    fname = "msleep.csv"
    names, body_brain = filter_data(*read_csv(fname))

    n = len(names)
    print(n)
    pearson_corr, p_val = stats.pearsonr(np.log(body_brain[:, 0]), np.log(body_brain[:, 1]))
    z_star = - stats.norm.ppf(0.05)
    c_minus = np.exp(-2 * z_star / np.sqrt(n)) * (1 + pearson_corr) / (1 - pearson_corr)
    c_plus = np.exp(+ 2 * z_star / np.sqrt(n)) * (1 + pearson_corr) / (1 - pearson_corr)
    conf_int = [(c_minus - 1) / (c_minus + 1), (c_plus - 1) / (c_plus + 1)]
    return conf_int


# print(pearson_corr_conf_int())


def lin_reg(plot=False):
    fname = "msleep.csv"
    names, body_brain = filter_data(*read_csv(fname))
    human_index = names.index("Human")
    remove_index = [human_index]    # Add other indices to filter out other mammals

    names_f = [name for i, name in enumerate(names) if i not in remove_index]   # f for filtered
    bb_log_f = np.array([np.log(pair) for i, pair in enumerate(body_brain) if i not in remove_index])

    avg_log_body = np.average(bb_log_f[:, 0])
    body_log_f_s = bb_log_f[:, 0] - avg_log_body    # s for shifted

    result = stats.linregress(body_log_f_s, bb_log_f[:, 1])
    slope, intercept = result.slope, result.intercept
    pearson_r, p_val = result.rvalue, result.pvalue
    slope_stderr, intercept_stderr = result.stderr, result.intercept_stderr
    print("Slope: {} with standard error {} (residual normality assumed)".format(slope, slope_stderr))
    print("Intercept: {} with standard error {} (residual normality assumed)".format(intercept, intercept_stderr))
    print("Pearson r: {}".format(pearson_r))
    print("p-value for Wald test (t-distribution) of null hypothesis of slope = 0: {}".format(p_val))

    if plot:
        def lin_func(x, slope, intercept):
            return slope * x + intercept

        fig, ax = plt.subplots(figsize=(3.0, 2.4))
        num_to_col = cm.ScalarMappable(norm=colors.Normalize(vmin=-0.5, vmax=1.5),
                                  cmap=colormaps["viridis"])
        color = num_to_col.to_rgba(1)
        color2 = num_to_col.to_rgba(-0.5)
        ax.scatter(bb_log_f[:, 0], bb_log_f[:, 1], color=color, s=10)
        ax.scatter(np.log(body_brain[human_index, 0]), np.log(body_brain[human_index, 1]), color="red", s=10, label="Human")
        linspace = np.linspace(min(body_log_f_s), max(body_log_f_s), 100)
        lin_reg_y = lin_func(linspace, slope, intercept)
        ax.plot(linspace + avg_log_body, lin_reg_y, color="black", label="Linear regression")

        log_mass = np.log(111.0)
        brain_est = lin_func(log_mass - avg_log_body, slope, intercept)
        plt.scatter(log_mass, brain_est, color=color2, s=10)
        def v_x(x):
            n = len(body_log_f_s)
            M_n = np.std(body_log_f_s)**2 * (n - 1)
            return 1 + 1 / n + x**2 / M_n

        sigma_hat = np.std(bb_log_f[:, 1] - lin_func(body_log_f_s, slope, intercept))
        t_l, t_u = stats.t(df=len(body_log_f_s) - 1).ppf((0.005, 0.995))
        f_upper = lin_reg_y + t_u * sigma_hat * np.sqrt(v_x(linspace))
        f_lower = lin_reg_y + t_l * sigma_hat * np.sqrt(v_x(linspace))
        ax.plot(linspace + avg_log_body, f_upper, color="red", linestyle="dashed")
        ax.plot(linspace + avg_log_body, f_lower, color="red", linestyle="dashed")

        ax.set_xlabel(r"log(body weight)")
        ax.set_ylabel(r"log(brain weight)")
        plt.tight_layout()
        plt.grid()
        plt.legend(handlelength=1.2)
        plt.savefig("Linear_regression_body_brain.svg")
        plt.show()

    return slope, intercept, pearson_r, p_val, slope_stderr, intercept_stderr, avg_log_body


lin_reg(plot=True)

def estimate_brain_size(body_weight):
    slope, intercept, _, _, _, _, avg_log_body = lin_reg(plot=False)
    brain_est_log = slope * (np.log(body_weight) - avg_log_body) + intercept
    return np.exp(brain_est_log)


# print(estimate_brain_size(111))

fname = "msleep.csv"
# names, body_brain = read_csv(fname)
# names_u, body_brain_u = filter_data(names, body_brain)
# print(names_u)
# print(body_brain_u)
