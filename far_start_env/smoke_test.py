"""Sanity checks + a first look at the alpha trade_off (not the final experiment)"""

import numpy as np
from envs import FarStartEnv
from learner import FilterLearner

env = FarStartEnv(T=50, k=4, n_actions=3, u=3, p_tremble=0.05)
pi_E = env.expert_policy()
uniform = np.full((env.nS, env.nA), 1 / env.nA)
print(f"Vmax = {env.vmax:.1f}   J(expert, tremble) = {env.evaluate(pi_E):.2f}   J(uniform) = {env.evaluate(uniform):.3f}")

budget, seeds = 400_000, 10
checkpoints = [2_000, 20_000, 100_000, 400_000]
anneal = lambda frac: max(0.0, 1.0-frac)
print(f"\nmean gap J(pi_E)-J(pi) at given #interactions ({seeds} seeds)")
print("alpha      " + "".join(f"{c:>10,}" for c in checkpoints) + "   off-support starts")

for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, anneal]:
    gaps, offs = [], []
    for seed in range(seeds):
        e = FarStartEnv(T=50, k=4, n_actions=3, u=3, p_tremble=0.05, seed=seed)
        log = FilterLearner(e, alpha=alpha, seed=seed).run(budget)
        idx = [min(np.searchsorted(log["interactions"], c), len(log["gap"]) - 1) for c in checkpoints]
        gaps.append(log["gap"][idx])
        offs.append(log["frac_offsupport_starts"][-1])
    name = "1->0" if callable(alpha) else f"{alpha:.2f}"
    print(f"{name:<11}" + "".join(f"{g:>10.2f}" for g in np.mean(gaps, axis=0)) + f"{np.mean(offs):>12.1%}")