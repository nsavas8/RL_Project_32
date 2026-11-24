"""
A2 vs theory: how does the final error grow with the horizon?
 
Theory (Swamy et al.):  Thm 3.8 / Cor. 4.1 say expert resets alone (alpha = 1) can give an error that grows
like T^2, while learning on your own distribution (alpha < 1) keeps it linear in T.
 
Mapping to our environment: the tremble p is a fixed per-step error rate (like epsilon in the theorems).
After training we split the total loss of a policy into two parts:
 
    Vmax - J(pi)  =  [Vmax - J(pi_E)]   +   [J(pi_E) - J(pi)]
     total loss      cost of slips when      extra cost of NOT knowing
                     you know how to          how to climb back
                     climb back (~ L)         (~ L^2 for alpha = 1, 0 for alpha < 1)
 
with L = T - k the corridor length (the part of the episode where slips happen; the lock adds a fixed offset).
Expected log-log slopes: total loss of alpha < 1 ~ 1, extra cost of alpha = 1 ~ 2.
 
We train real learners (not the predicted policies) and compare with the exact prediction, which is
computed from the policy each learner should converge to.
 
Run:  python3 t_scaling.py      (about 5 min on 2 cores)
Out:  results/t_scaling.npz, results/fig_t_scaling.png, printed table
"""
import os
import time
import numpy as np
from multiprocessing import Pool
from envs import FarStartEnv
from learner import FilterLearner
import matplotlib
import matplotlib.pyplot as plt

ENV = dict(k=4, n_actions=3, u=3, p_tremble=0.05)
T_LIST = [10, 20, 30, 50, 70]
ALPHAS = [1.0, 0.5]
SEEDS = 20

def budget(T):
    return 15_000 * T


def predicted_policy(env, alpha):
    # The policy each learner converges to
    pi = env.expert_policy()
    if alpha == 1.0:
        for s in env.OFF.values():
            pi[s] = 1.0 / env.nA
    return pi


def one_run(job):
    T, alpha, seed = job
    env = FarStartEnv(T=T, **ENV, seed=seed)
    L = FilterLearner(env, alpha=alpha, seed=seed)
    L.run(budget(T))
    return T, alpha, seed, env.evaluate(L.pi)

def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]



if __name__ == "__main__":
    jobs = [(T, a, s) for T in T_LIST for a in ALPHAS for s in range(SEEDS)]
    J = np.zeros((len(T_LIST), len(ALPHAS), SEEDS))

    t0 = time.time()
    with Pool() as pool:
        for i, (T, a, s, j) in enumerate(pool.imap_unordered(one_run, jobs), 1):
            J[T_LIST.index(T), ALPHAS.index(a), s] = j
            print(f"\r{i}/{len(jobs)} runs ({time.time() - t0:.0f} s)", end="", flush=True)

    Lc = np.array([T - ENV["k"] for T in T_LIST])
    vmax = np.array([FarStartEnv(T=T, **ENV).vmax for T in T_LIST])
    JE = np.array([FarStartEnv(T=T, **ENV).evaluate(FarStartEnv(T=T, **ENV).expert_policy()) for T in T_LIST])
    pred = np.array([[FarStartEnv(T=T, **ENV).evaluate(predicted_policy(FarStartEnv(T=T, **ENV), a)) for a in ALPHAS] for T in T_LIST])
    converged = np.abs(J - pred[:, :, None]) < 1e-6
    i1, i5 = ALPHAS.index(1.0), ALPHAS.index(0.5)
    extra1 = JE -J[:, i1].mean(axis=1)
    total5 = vmax - J[:, i5].mean(axis=1)
    total1 = vmax - J[:, i1].mean(axis=1)
    print(f"\n{'T':>3} {'L':>3} | {'alpha=1: JE-J':>14} {'pred':>7} | {'alpha=0.5: Vmax-J':>18} {'pred':>7} | converged a=1 / a=0.5")
    for n, T in enumerate(T_LIST):
        print(f"{T:>3} {Lc[n]:>3} | {extra1[n]:>14.3f} {JE[n] - pred[n, i1]:>7.3f} | {total5[n]:>18.3f} {vmax[n] - pred[n, i5]:>7.3f} |"
              f" {converged[n, i1].mean():>6.0%} / {converged[n, i5].mean():.0%}")
    print(f"\nlog-log slope vs L:  alpha=1 extra cost {slope(Lc, extra1):.2f} (theory 2)   "
          f"alpha=0.5 total loss {slope(Lc, total5):.2f} (theory 1)   alpha=1 total loss {slope(Lc, total1):.2f} (mix of both)")
 
    os.makedirs("results", exist_ok=True)
    np.savez("results/t_scaling.npz", T=T_LIST, L=Lc, alphas=ALPHAS, J=J, J_expert=JE, vmax=vmax, J_pred=pred)

    
    # Figure
    matplotlib.use("Agg")
    
    INK, INK2 = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    C1, C5 = "#104281", "#2a78d6"
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(Lc, total1, "o-", color=C1, lw=1.8, ms=7, label="α = 1: total loss  Vmax − J(π)")
    ax.plot(Lc, extra1, "s-", color=C1, lw=1.8, ms=7, mfc="#fcfcfb", label="α = 1: extra cost  J(π_E) − J(π)")
    ax.plot(Lc, total5, "o-", color=C5, lw=1.8, ms=7, label="α = 0.5: total loss  Vmax − J(π)  (extra cost = 0)")
    for p_, x0, y0, txt in [(2, Lc[0], extra1[0], "slope 2"), (1, Lc[0], total5[0], "slope 1")]:
        xs = np.array([Lc[0], Lc[-1]])
        ax.plot(xs, y0 * (xs / x0) ** p_, ":", color=INK2, lw=1)
        ax.text(xs[1] * 1.04, y0 * (xs[1] / x0) ** p_, txt, color=INK2, fontsize=9, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(Lc, [str(v) for v in Lc]); ax.minorticks_off()
    ax.set_xlim(Lc[0] * 0.85, Lc[-1] * 1.35)
    ax.set_xlabel("corridor length L = T − k  (log)")
    ax.set_ylabel("loss in reward  (log)")
    ax.set_title(f"Error vs horizon after training (p = {ENV['p_tremble']}, mean of {SEEDS} seeds)",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig("results/fig_t_scaling.png", dpi=200)
    print("saved results/t_scaling.npz, results/fig_t_scaling.png")
 
