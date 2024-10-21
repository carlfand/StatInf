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
    print(pearson_corr)
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
    plt.savefig("body_vs_brain_weight.svg")
    plt.show()

plot_filtered_data()


fname = "msleep.csv"
# names, body_brain = read_csv(fname)
# names_u, body_brain_u = filter_data(names, body_brain)
# print(names_u)
# print(body_brain_u)
