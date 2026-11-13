"""
Plots and summary table for A2b, from results/sweep.npz (run sweep.py first).
 
  results/fig_cost_vs_error.png  gap vs interactions per alpha, mean with 95% bootstrap CI   (report Fig. 8a)
  results/fig_best_alpha.png     mean gap for every (alpha, budget), best alpha marked        (report Fig. 8b)
  results/summary.txt            gap at checkpoints + interactions needed to reach a target gap
 
Run:  python3 plots.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
 
D = np.load("results/sweep.npz")
grid, names, vmax = D["grid"], list(D["alpha_names"]), float(D["vmax"])
gaps = D["gaps"] / vmax                         # normalised gap: 0 = expert, ~0.79 = random policy
n_seeds = gaps.shape[1]
rng = np.random.default_rng(0)
 
# ---------- colours (validated: ordinal blue ramp for fixed alpha, categorical orange for the schedule) ----------
COLOR = {"0": "#86b6ef", "0.25": "#5598e7", "0.5": "#2a78d6", "0.75": "#1c5cab", "1": "#104281", "1→0": "#eb6834"}
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDC,
    "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})
 
 
def boot_ci(x, n=2000):
    """95% bootstrap CI of the mean over seeds (axis 0)."""
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)
 
 
# ---------- Fig. 8a: cost vs error ----------
fig, ax = plt.subplots(figsize=(7.5, 4.3))
for a, name in enumerate(names):
    m = gaps[a].mean(axis=0)
    lo, hi = boot_ci(gaps[a])
    ls = "--" if name == "1→0" else "-"
    ax.fill_between(grid, lo, hi, color=COLOR[name], alpha=0.15, linewidth=0)
    ax.plot(grid, m, ls, color=COLOR[name], linewidth=1.8, label=f"α = {name}" if name != "1→0" else "α: 1 → 0 (annealed)")
ax.set_xscale("log")
ax.set_xlim(grid[0], grid[-1])
ax.set_ylim(-0.01, 0.82)
ax.set_xlabel("environment interactions")
ax.set_ylabel("gap  (J(π_E) − J(π)) / Vmax")
ax.set_title("Far-start environment: gap to expert vs. interactions\n"
             f"T = {int(D['env_T'])}, k = {int(D['env_k'])}, |A| = {int(D['env_n_actions'])}, "
             f"u = {int(D['env_u'])}, p = {float(D['env_p_tremble'])}; mean of {n_seeds} seeds, 95% CI",
             loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("results/fig_cost_vs_error.png", dpi=200)
 
# ---------- Fig. 8b: best alpha vs budget ----------
fixed = [i for i, n in enumerate(names) if n != "1→0"]
mean = gaps[fixed].mean(axis=1)                                  # (n_alpha, n_grid)
best_val = mean.min(axis=0)
TIE = 0.02                                                       # within 2% of Vmax of the best counts as a tie
learned = best_val < 0.7                                         # before this, no alpha has learned anything: no "best"
near_best = (mean <= best_val + TIE) & learned
 
ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("blue_seq", ramp)
edges_x = np.sqrt(grid[:-1] * grid[1:])
edges_x = np.concatenate([[grid[0] ** 2 / edges_x[0]], edges_x, [grid[-1] ** 2 / edges_x[-1]]])
edges_y = np.arange(len(fixed) + 1) - 0.5
TOP = len(fixed) - 0.5
 
fig, ax = plt.subplots(figsize=(7.5, 3.9))
pc = ax.pcolormesh(edges_x, edges_y, mean, cmap=cmap, vmin=0, vmax=0.8, shading="flat")
yy, xx = np.nonzero(near_best)
ax.scatter(grid[xx], yy, s=7, color="#f2a33a", edgecolor="none", label="best α (ties: within 0.02·Vmax of the best)")
first = grid[np.argmax(learned)]
ax.text(np.sqrt(grid[0] * first), len(fixed) - 0.5 + 0.15, "no α has learned yet", ha="center",
        va="bottom", fontsize=8.5, color=INK2)
ax.axvline(first, color=INK2, linewidth=0.8, linestyle=":")
ax.set_xscale("log")
ax.set_yticks(range(len(fixed)), [f"α = {names[i]}" for i in fixed])
ax.set_ylim(-0.5, TOP + 0.55)
ax.set_xlabel("interaction budget")
ax.grid(False)
ax.set_title("Which α is best for a given budget?  (colour = mean gap / Vmax, dots = best α)",
             loc="left", fontsize=10, color=INK)
cb = fig.colorbar(pc, ax=ax, pad=0.02)
cb.set_label("mean gap / Vmax")
cb.outline.set_visible(False)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22), fontsize=9)
fig.tight_layout()
fig.savefig("results/fig_best_alpha.png", dpi=200)
 
# ---------- summary table ----------
def gap_at(b):
    return np.searchsorted(grid, b, side="right") - 1
 
lines = [f"Far-start sweep: {n_seeds} seeds, Vmax = {vmax:.0f}. Gap as fraction of Vmax, mean [95% CI].", ""]
checkpoints = [5_000, 20_000, 100_000, 400_000]
lines.append("alpha    " + "".join(f"{c:>22,}" for c in checkpoints) + "   off-support starts")
for a, name in enumerate(names):
    row = f"{name:<9}"
    for c in checkpoints:
        j = gap_at(c)
        lo, hi = boot_ci(gaps[a][:, [j]])
        row += f"{gaps[a][:, j].mean():>8.3f} [{lo[0]:.3f},{hi[0]:.3f}]"
    lines.append(row + f"{D['offsupport'][a].mean():>12.1%}")
 
for target in [0.5, 0.05]:
    lines += ["", f"Interactions until gap <= {target}·Vmax (first time; grid resolution ~6%): "
                  "median [25%, 75%] over seeds that reach it, and share of seeds that reach it"]
    for a, name in enumerate(names):
        hit = gaps[a] <= target
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
print("\nsaved results/fig_cost_vs_error.png, results/fig_best_alpha.png, results/summary.txt")