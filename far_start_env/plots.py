"""
Figure 8 (A2b) and summary table, from results/sweep.npz (run sweep.py first).
Style matches Jana's figures (default matplotlib, two panels side by side, dpi 150).

  results/fig8_tradeoff.png   (a) gap vs interactions per alpha, (b) best alpha vs budget
  results/summary.txt         gap at checkpoints + interactions needed to reach a target gap

Run:  python3 plots.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU = 0.05            # target gap / Vmax (dotted line in (a), strict target in the table)
TIE = 0.02            # in (b): alphas within TIE of the best mean gap count as "also best"
LEARNED = 0.7         # in (b): only budgets where the best alpha is below this (before, nobody has learned)

D = np.load("results/sweep.npz")
grid, names, vmax = D["grid"], list(D["alpha_names"]), float(D["vmax"])
gaps = D["gaps"] / vmax                       # shape (n_alpha, n_seeds, n_grid)
n_seeds = gaps.shape[1]
rng = np.random.default_rng(0)


def boot_ci(x, n=2000):
    """95% bootstrap CI of the mean over seeds (axis 0)."""
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))

# ---------- (a) gap vs interactions ----------
for i, name in enumerate(names):
    m = gaps[i].mean(axis=0)
    lo, hi = boot_ci(gaps[i])
    annealed = name == "1→0"
    label = r"$\alpha$: 1 → 0 (annealed)" if annealed else rf"$\alpha={name}$"
    ax[0].plot(grid, m, "--" if annealed else "-", color=f"C{i}", label=label)
    ax[0].fill_between(grid, lo, hi, color=f"C{i}", alpha=0.15)
ax[0].axhline(TAU, color="k", ls=":", lw=0.8)
ax[0].set_xscale("log")
ax[0].set_xlabel("environment interactions")
ax[0].set_ylabel("gap / V_max")
ax[0].set_title(f"Far start (T = {int(D['env_T'])}): gap vs. interactions")
ax[0].legend(fontsize=8)

# ---------- (b) best alpha vs budget ----------
fixed = [i for i, n in enumerate(names) if n != "1→0"]
alpha_vals = np.array([float(names[i]) for i in fixed])
mean = gaps[fixed].mean(axis=1)                       # (n_fixed_alpha, n_grid)
best = mean.min(axis=0)
budgets = np.unique(np.geomspace(grid[0], grid[-1], 25).astype(int))   # fewer points than the grid, readable
xs, y_best, y_lo, y_hi = [], [], [], []
for b in budgets:
    j = np.searchsorted(grid, b, side="right") - 1
    if best[j] >= LEARNED:
        continue
    tied = alpha_vals[mean[:, j] <= best[j] + TIE]
    xs.append(grid[j])
    y_best.append(alpha_vals[np.argmin(mean[:, j])])
    y_lo.append(tied.min())
    y_hi.append(tied.max())
xs, y_best, y_lo, y_hi = map(np.array, (xs, y_best, y_lo, y_hi))
ax[1].fill_between(xs, y_lo, y_hi, color="C0", alpha=0.15, step="mid",
                   label=rf"$\alpha$ within {TIE} of the best")
ax[1].plot(xs, y_best, "o-", color="C0", drawstyle="steps-mid", label=r"best $\alpha$ (lowest mean gap)")
ax[1].set_xscale("log")
ax[1].set_xlim(grid[0], grid[-1])
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_yticks(alpha_vals)
ax[1].set_xlabel("interaction budget")
ax[1].set_ylabel(r"$\alpha$")
ax[1].set_title(r"Best $\alpha$ vs. budget")
ax[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/fig8_tradeoff.png", dpi=150)

# ---------- summary table ----------
def col(b):
    return np.searchsorted(grid, b, side="right") - 1

lines = [f"Far-start sweep: {n_seeds} seeds, Vmax = {vmax:.0f}. Gap as fraction of Vmax, mean [95% CI].", ""]
checkpoints = [5_000, 6_000, 10_000, 20_000, 50_000, 100_000, 200_000, 400_000]
lines.append("alpha    " + "".join(f"{c:>22,}" for c in checkpoints) + "   off-support starts")
for i, name in enumerate(names):
    row = f"{name:<9}"
    for c in checkpoints:
        j = col(c)
        lo, hi = boot_ci(gaps[i][:, [j]])
        row += f"{gaps[i][:, j].mean():>8.3f} [{lo[0]:.3f},{hi[0]:.3f}]"
    lines.append(row + f"{D['offsupport'][i].mean():>12.1%}")

for target in [0.5, TAU]:
    lines += ["", f"Interactions until gap <= {target}·Vmax (first time; grid resolution ~6%): "
                  "median [25%, 75%] over seeds that reach it, and share of seeds that reach it"]
    for i, name in enumerate(names):
        hit = gaps[i] <= target
        reached = hit.any(axis=1)
        cost = grid[np.argmax(hit, axis=1)][reached]
        if len(cost):
            q = np.percentile(cost, [25, 50, 75])
            lines.append(f"  α = {name:<5} {q[1]:>9,.0f}  [{q[0]:>7,.0f}, {q[2]:>7,.0f}]   reached: {reached.mean():.0%}")
        else:
            lines.append(f"  α = {name:<5}      never                         reached: 0%")

text = "\n".join(lines)
open("results/summary.txt", "w").write(text + "\n")
print(text)
print("\nsaved results/fig_tradeoff.png, results/summary.txt")