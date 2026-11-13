"""
Full alpha sweep for A2b: 6 alpha settings x 20 seeds on the far-start environment.

Every run logs its gap after each iteration, but iterations end at different
interaction counts in different runs. To average over seeds, each run's curve is
read off on one common grid of budgets: "gap of the policy in use after b
interactions". The result is saved to results/sweep.npz, so plots.py can be
re-run without training again.

Run:  python3 sweep.py      (about 2 to 5 min, uses all CPU cores)
"""
import os
import time
import numpy as np
from multiprocessing import Pool
from envs import FarStartEnv
from learner import FilterLearner

ENV = dict(T=50, k=4, n_actions=3, u=3, p_tremble=0.05)
BUDGET = 400_000
SEEDS = 20
GRID = np.unique(np.geomspace(500, BUDGET, 120).astype(int))   # budgets at which we read off the gap


def anneal(frac):
    """alpha from 1 down to 0 over the budget. A named function (not a lambda) so worker processes can use it."""
    return max(0.0, 1.0 - frac)


ALPHAS = {"0": 0.0, "0.25": 0.25, "0.5": 0.5, "0.75": 0.75, "1": 1.0, "1→0": anneal}


def on_grid(log, gap_start):
    """For each budget b in GRID: gap of the last policy reached with <= b interactions."""
    idx = np.searchsorted(log["interactions"], GRID, side="right") - 1
    return np.where(idx >= 0, log["gap"][np.maximum(idx, 0)], gap_start)


def one_run(job):
    name, seed = job
    env = FarStartEnv(**ENV, seed=seed)
    uniform = np.full((env.nS, env.nA), 1 / env.nA)
    gap_start = env.evaluate(env.expert_policy()) - env.evaluate(uniform)   # gap before any update
    log = FilterLearner(env, alpha=ALPHAS[name], seed=seed).run(BUDGET)
    return name, seed, on_grid(log, gap_start), log["frac_offsupport_starts"][-1]


if __name__ == "__main__":
    names = list(ALPHAS)
    jobs = [(n, s) for n in names for s in range(SEEDS)]
    gaps = np.zeros((len(names), SEEDS, len(GRID)))
    offsupport = np.zeros((len(names), SEEDS))

    t0 = time.time()
    with Pool() as pool:
        for i, (name, seed, g, off) in enumerate(pool.imap_unordered(one_run, jobs), 1):
            a = names.index(name)
            gaps[a, seed], offsupport[a, seed] = g, off
            print(f"\r{i}/{len(jobs)} runs done ({time.time() - t0:.0f} s)", end="", flush=True)

    env = FarStartEnv(**ENV)
    os.makedirs("results", exist_ok=True)
    np.savez(
        "results/sweep.npz",
        grid=GRID, gaps=gaps, offsupport=offsupport, alpha_names=np.array(names),
        vmax=env.vmax, J_expert=env.evaluate(env.expert_policy()), budget=BUDGET,
        **{f"env_{k}": v for k, v in ENV.items()},
    )
    print("\nsaved results/sweep.npz")