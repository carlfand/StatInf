import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.special import comb		# comb(n, k) is n choose k.

rcParams.update({"text.usetex": True})
plt.rc("text.latex")

"""We have shown i problem 2.18 (2.19 October version) that B_n(p) [the Bernstein polynomials] converges in probability towards g(p),
g: [0, 1] -> R, continuous, and B_n(p) is defined by X_n ~ Binom(n, p), B_n(p) = 
E_p[g(X_n / n)] = sum g(j / n) n choose j p^j(1 - p)^(n - j).
Here, we visualise this convergence for a specific function.
"""

def g_1(p):
	"""g_1: [0, 1] -> R, uniformly continuous (and ugly-looking)."""
	return np.sin(2 * np.pi * p) + np.exp(1.234 * np.sin(np.sqrt(p))**3) - np.exp(-4.321 * np.cos(p**2)**5)


def binomial_density(n, p, k):
	"""Probability density of Binom(n, p) at k."""
	return comb(n, k) * p ** k * (1 - p) ** (n - k)


def bernstein_n(n, p, func):
	b_n = 0.0
	for k in range(n + 1):	# Including n.
		b_n += binomial_density(n, p, k) * func(k / n)
	return b_n

def plot_approximations(func, *ns):
	"""Approximating function func with bernstein polynomials of degree n."""
	fig, ax = plt.subplots(figsize=(6, 4))
	x_axis = np.linspace(0, 1, 100, endpoint=True)
	plt.plot(x_axis, func(x_axis), color="black", linewidth=1.5, label=r"$f(x)$")
	for n in ns:
		bernstein_approx = bernstein_n(n, x_axis, func)
		ax.plot(x_axis, bernstein_approx, linestyle="dashed", linewidth=1.0, color="blue", label=r"$B_{}(x)$".format("{" + str(n) + "}"))
	ax.legend(handlelength=1.2)
	ax.set_xlim(0, 1)
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$f(x)$")
	plt.show()


plot_approximations(g_1, 10, 20, 40, 80)
