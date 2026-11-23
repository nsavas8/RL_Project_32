"""
5.3: error vs horizon T on DANTE.  Learner A (true reward) and learner B (learned discriminator), alpha in {0, 0.5, 1},
plus sampled MMDP (F closed under negation = 'sym', F={r} = 'one') with the same interaction budget.
Run:   python dante_sweep.py --quick     (about 1 min, sanity check)
       python dante_sweep.py             (full)
Outputs (folder results_dante/): raw_runs.csv, summary.csv, gap_vs_T.png
Gap = J* - J(pi) with J* the optimal value WITH tremble (exact DP), normalised by V_max = J(expert, no tremble) = T.
We also store the gap to the expert (J(pi_E) - J(pi)) in the csv. Metrics per run: final gap / V_max at the budget;
first interaction count at which gap <= TAU * V_max (nan = never).
"""
import argparse, csv, math, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from learner import FilterLearner
from disc_learner import DiscFilterLearner
from dante_port import DanteEnv, run_mmdp

TAU, P_TREMBLE, N_DEMOS = 0.05, 0.05, 20
ALPHAS = [0.0, 0.5, 1.0]
T_LIST, BUDGETS, N_SEEDS = [10, 20, 50, 100], [5_000, 20_000], 20


def job(a):
    method, T, B, alpha, seed = a
    env = DanteEnv(T, P_TREMBLE, seed); vmax = env.vmax
    if method in ("A", "B"):
        L = (FilterLearner(env, alpha=alpha, seed=seed) if method == "A"
             else DiscFilterLearner(env, alpha=alpha, seed=seed, n_demos=N_DEMOS))
        log = L.run(B)                                       # log["gap"] is J(pi_E) - J(pi) (learner.py is unchanged)
        off = env.optimal_value() - env.expert_value()       # J* - J(pi_E)  ->  gap to J* = gap to expert + off
        g_opt, g_exp = (log["gap"] + off) / vmax, log["gap"] / vmax
        hit = np.where(g_opt <= TAU)[0]
        return a, g_opt[-1], (log["interactions"][hit[0]] if len(hit) else float("nan")), g_exp[-1]
    M = max(1, B // (T * (T + 1) // 2))
    n, gap_opt, gap_exp = run_mmdp(T, M, method, seed, P_TREMBLE)       # method = 'sym' or 'one'
    return a, gap_opt / vmax, (n if gap_opt / vmax <= TAU else float("nan")), gap_exp / vmax


def boot(x, n=2000):
    x = np.asarray(x, float); rng = np.random.default_rng(0)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); q = ap.parse_args().quick
    Ts, Bs, S = ([10, 50], [5_000], 3) if q else (T_LIST, BUDGETS, N_SEEDS)
    jobs = [(m, T, B, al, s) for T in Ts for B in Bs for s in range(S)
            for m, al in [("A", a) for a in ALPHAS] + [("B", a) for a in ALPHAS] + [("sym", 0), ("one", 0)]]
    os.makedirs("results_dante", exist_ok=True)
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(job, jobs, chunksize=4))
    with open("results_dante/raw_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "T", "budget", "alpha", "seed", "gap_to_opt_over_vmax", "cost_to_target", "gap_to_expert_over_vmax"])
        for (m, T, B, al, s), g, c, ge in res: w.writerow([m, T, B, al, s, g, c, ge])
    groups = {}
    for (m, T, B, al, s), g, c, ge in res: groups.setdefault((m, al if m in "AB" else "-", T, B), []).append((g, c, ge))
    with open("results_dante/summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "alpha", "T", "budget", "mean_gap_to_opt", "ci_lo", "ci_hi", "frac_reached_target", "mean_gap_to_expert"])
        for k, v in sorted(groups.items(), key=str):
            m_, lo, hi = boot([x[0] for x in v])
            w.writerow([*k, f"{m_:.4f}", f"{lo:.4f}", f"{hi:.4f}", np.mean([not math.isnan(x[1]) for x in v]), f"{np.mean([x[2] for x in v]):.4f}"])
            print(*k, f"gap to J*/Vmax = {m_:.3f} [{lo:.3f}, {hi:.3f}]  reached tau: {np.mean([not math.isnan(x[1]) for x in v]):.2f}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(Bs), figsize=(6 * len(Bs), 4), squeeze=False)
    for j, B in enumerate(Bs):
        for (m, al) in [("A", a) for a in ALPHAS] + [("B", a) for a in ALPHAS] + [("sym", "-"), ("one", "-")]:
            al_k = al if m in ("A", "B") else "-"
            y = [boot([x[0] for x in groups[(m, al_k, T, B)]]) for T in Ts]
            lab = f"{'A true reward' if m=='A' else 'B discriminator'} α={al}" if m in "AB" else f"MMDP {m}"
            ax[0, j].plot(Ts, [v[0] for v in y], "o-" if m != "B" else "s--", label=lab)
        ax[0, j].set_xlabel("horizon T"); ax[0, j].set_ylabel("(J* - J) / V_max"); ax[0, j].set_title(f"DANTE, budget {B}")
        ax[0, j].axhline(TAU, color="k", ls=":", lw=0.8)
    ax[0, 0].legend(fontsize=6); plt.tight_layout(); plt.savefig("results_dante/gap_vs_T.png", dpi=150)
