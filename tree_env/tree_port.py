"""
Tree experiments (A2a) rewritten to use the shared FILTER learner of the far-start code (learner.py, unchanged).

Files needed in the same folder:  learner.py  (her FilterLearner, NOT modified)

What is in here
---------------
TreeEnv              the Thm-3.1 tree with the SAME interface as her FarStartEnv:
                     reset(), reset_to(s, t), step(a) -> (s, r, done), expert_state(t), expert_policy(),
                     expert_support(), evaluate(pi), n_interactions, nS, nA, T, vmax.
                     Interactions = step() calls. reset / reset_to / evaluate are free (same rule as hers).
SparseFilterLearner  her learner with dict storage instead of dense (nS x nA) arrays. Same update rule, same RNG
                     calls, so it gives IDENTICAL results to her dense learner (checked by --validate).
                     Needed because the tree has 2^(T+1)-1 states (dense arrays do not fit at T = 50).
run_mmdp             sampled MMDP (Alg. 3) on the same environment, same interaction counting.

Usage
-----
python tree_port.py --validate      sparse == dense check on small trees (do this first)
python tree_port.py --quick         tiny sweep (a minute) to see that everything runs
python tree_port.py                 full sweep
"""
import argparse
import csv
import math
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from learner import FilterLearner          # her learner, unchanged

# ============================== configuration ==============================
A = 2
TAU = 0.5                    # target: gap <= TAU * Vmax  (Thm 3.1 uses 0.5)
ROLLOUTS_PER_ITER = 10       # M of her learner (default in learner.py); the target is checked after each batch
COST_CAP = 20_000_000
DELTA = 0.1
N_SEEDS = 20
ALPHAS_FULL = [0.0, 0.5, 1.0]
ALPHAS_LONG = [0.5, 1.0]
T_FULL = [2, 4, 6, 8, 10, 12, 14, 16]
T_LONG = [20, 30, 40, 50]
M_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
MMDP_SUCCESS_RATE = 0.9


# ============================== environment ==============================
class TreeEnv:
    """Complete A-ary tree of depth T. State = node id (root 0, children of h are A*h+1+a). Time t = depth (0..T).
    Episode = T steps from the root to a leaf. Reward 1 at the expert's leaf, 0 elsewhere. Expert: always action 0."""

    def __init__(self, T, n_actions=2):
        self.T, self.nA = T, n_actions
        self.nS = (n_actions ** (T + 1) - 1) // (n_actions - 1)
        self.expert = [0]
        for _ in range(T):
            self.expert.append(self.nA * self.expert[-1] + 1)
        self.vmax = 1.0
        self.n_interactions = 0
        self.s, self.t = 0, 0

    def reset(self):
        self.s, self.t = 0, 0
        return self.s

    def reset_to(self, s, t):                       # free: expert states are known from the demonstrations
        self.s, self.t = s, t
        return self.s

    def step(self, a):                              # the ONLY place where interactions are counted
        self.n_interactions += 1
        self.s = self.nA * self.s + 1 + a
        self.t += 1
        done = self.t == self.T
        r = 1.0 if (done and self.s == self.expert[self.T]) else 0.0
        return self.s, r, done

    def expert_state(self, t):
        return self.expert[t]

    def expert_support(self):
        return set(self.expert)

    def expert_policy(self):                        # dense, only for small T (validation)
        pi = np.full((self.nS, self.nA), 1.0 / self.nA)
        for d in range(self.T):
            pi[self.expert[d]] = 0.0
            pi[self.expert[d], 0] = 1.0
        return pi

    def evaluate(self, pi):
        """Exact J(pi) = P(policy reaches the expert's leaf). pi[s] must return a probability vector. Free."""
        p = 1.0
        for d in range(self.T):
            p *= pi[self.expert[d]][0]
        return p


# ============================== sparse version of her learner ==============================
class SparsePolicy:
    def __init__(self, nA):
        self.d, self.u = {}, np.full(nA, 1.0 / nA)

    def __getitem__(self, s):
        return self.d.get(s, self.u)


class SparseFilterLearner(FilterLearner):
    """Same rule as FilterLearner (mean Q_hat per (s,a), argmax, uniform over ties; unseen states uniform),
    stored in dicts so that only visited states cost memory. rollout() and act() are inherited unchanged."""

    def __init__(self, env, alpha=0.5, rollouts_per_iter=10, seed=0):
        self.env, self.alpha, self.M = env, alpha, rollouts_per_iter
        self.rng = np.random.default_rng(seed)
        self.q_sum, self.q_cnt = defaultdict(float), defaultdict(float)
        self.pi, self._touched = SparsePolicy(env.nA), set()

    def rollout(self, alpha):
        s_t, expert_reset = super().rollout(alpha)
        self._touched.add(s_t)
        return s_t, expert_reset

    def update_policy(self):
        for s in self._touched:
            mean = np.array([self.q_sum[(s, a)] / self.q_cnt[(s, a)] if self.q_cnt[(s, a)] > 0 else 0.0
                             for a in range(self.env.nA)])
            best = mean == mean.max()
            self.pi.d[s] = best / best.sum()
        self._touched.clear()


# ============================== cost to target ==============================
def cost_to_target(T, alpha, seed, M=ROLLOUTS_PER_ITER, tau=TAU, cap=COST_CAP, learner_cls=SparseFilterLearner):
    """Interactions until gap = J(pi_E) - J(pi) <= tau * Vmax, checked after every batch of M rollouts. None = cap hit."""
    env = TreeEnv(T, A)
    L = learner_cls(env, alpha=alpha, rollouts_per_iter=M, seed=seed)
    while env.n_interactions < cap:
        for _ in range(L.M):
            L.rollout(alpha)
        L.update_policy()
        if 1.0 - env.evaluate(L.pi) <= tau * env.vmax:
            return env.n_interactions
    return None


# ============================== sampled MMDP (Alg. 3) ==============================
def run_mmdp(T, M, rng):
    """For t = T-1..0: M samples = reset to the expert state at time t (free), uniformly random first action,
    then the already learned pi_{t+1:T-1} (uniform off the expert's path). Importance weight |A|.
    pi_t minimises |Q_hat(expert action) - Q_hat(a)| (moment matching with f = +- leaf indicator), ties uniform.
    Cost is exactly M*T(T+1)/2 interactions."""
    env, policy = TreeEnv(T, A), {}
    for t in range(T - 1, -1, -1):
        ret = [0.0] * A
        for _ in range(M):
            a0 = rng.randrange(A)
            env.reset_to(env.expert_state(t), t)
            s, R, done = env.step(a0)
            while not done:
                if s == env.expert[env.t] and env.t in policy:
                    a = rng.choices(range(A), weights=policy[env.t])[0]
                else:
                    a = rng.randrange(A)
                s, r, done = env.step(a)
                R += r
            ret[a0] += R
        qhat = [A * ret[a] / M for a in range(A)]
        loss = [abs(qhat[0] - qhat[a]) for a in range(A)]
        best = [a for a in range(A) if loss[a] <= min(loss) + 1e-12]
        policy[t] = [1.0 / len(best) if a in best else 0.0 for a in range(A)]
    p_reach = math.prod(policy[d][0] for d in range(T))
    return env.n_interactions, (1.0 - p_reach) <= TAU * env.vmax


# ============================== bounds (constants = 1, Sec. 4.4 / 4.5 of the report) ==============================
def bound_M(T, delta=DELTA):
    eps = 1.0 / (2 * T)
    return math.log(A ** (2 * T) * T / delta) * (T * A) ** 2 / eps ** 2


def bound_total(T, delta=DELTA):
    return T * T * bound_M(T, delta)


def bound_crossover(Tmax=300):
    ok = None
    for T in range(2, Tmax + 1):
        ok = (ok or T) if bound_total(T) < A ** T else None
    return ok


# ============================== validation ==============================
def validate():
    """Sparse learner must equal her dense learner: same interactions and same policy on every state."""
    bad = 0
    for T in (3, 4, 5, 6):
        for alpha in (0.0, 0.5, 1.0):
            for seed in range(5):
                envs, learners = [], []
                for cls in (FilterLearner, SparseFilterLearner):
                    e = TreeEnv(T, A)
                    L = cls(e, alpha=alpha, rollouts_per_iter=10, seed=seed)
                    for _ in range(100):
                        for _ in range(L.M):
                            L.rollout(alpha)
                        L.update_policy()
                    envs.append(e); learners.append(L)
                same_n = envs[0].n_interactions == envs[1].n_interactions
                same_pi = all(np.array_equal(learners[0].pi[s], learners[1].pi[s]) for s in range(envs[0].nS))
                if not (same_n and same_pi):
                    bad += 1
                    print(f"MISMATCH T={T} alpha={alpha} seed={seed}")
    # environment sanity checks
    e = TreeEnv(4, A)
    assert e.evaluate(e.expert_policy()) == 1.0 and abs(e.evaluate(np.full((e.nS, 2), 0.5)) - 0.5 ** 4) < 1e-12
    e.reset(); n0 = e.n_interactions
    for _ in range(4):
        s, r, done = e.step(0)
    assert done and r == 1.0 and e.n_interactions - n0 == 4
    for T in (4, 8):
        n, ok = run_mmdp(T, 6, random.Random(0))
        assert n == 6 * T * (T + 1) // 2, (n, T)
    print("validation:", "OK (sparse == dense, env checks, MMDP cost formula)" if bad == 0 else f"{bad} mismatches")


# ============================== driver ==============================
def _job(task):
    kind, T, param, s = task
    if kind == "alpha":
        return kind, T, param, s, cost_to_target(T, param, seed=[T, int(round(param * 100)), s]), None
    steps, ok = run_mmdp(T, param, random.Random(f"mmdp-{T}-{param}-{s}"))
    return kind, T, param, s, steps, ok


def bootstrap_ci(x, n=2000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results_port")
    args = ap.parse_args()
    if args.validate:
        return validate()
    t_full, t_long, seeds = (T_FULL, T_LONG, N_SEEDS)
    if args.quick:
        t_full, t_long, seeds = [2, 4, 6, 8], [20], 4
    os.makedirs(args.out, exist_ok=True)
    tasks = []
    for T in t_full:
        tasks += [("alpha", T, al, s) for al in ALPHAS_FULL for s in range(seeds)]
    for T in t_long:
        tasks += [("alpha", T, al, s) for al in ALPHAS_LONG for s in range(seeds)]
    for T in t_full + t_long:
        tasks += [("mmdp", T, M, s) for M in M_GRID for s in range(seeds)]
    tasks.sort(key=lambda t: -(t[1] if t[0] == "alpha" and t[2] == 0.0 else 0))   # slowest jobs first
    with ProcessPoolExecutor() as ex:
        raw = list(ex.map(_job, tasks, chunksize=1))
    with open(os.path.join(args.out, "raw_runs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "T", "param", "seed", "interactions", "reached_target"])
        for kind, T, param, s, steps, ok in raw:
            w.writerow([kind, T, param, s, "" if steps is None else steps, "" if ok is None else int(ok)])

    cost, succ = {}, {}
    for kind, T, param, s, steps, ok in raw:
        if kind == "alpha":
            cost.setdefault((T, param), []).append(steps)
        else:
            succ.setdefault((T, param), []).append(bool(ok))
    all_T = sorted(set(t_full + t_long))
    mstar = {T: next((M for M in M_GRID if np.mean(succ[(T, M)]) >= MMDP_SUCCESS_RATE), None) for T in all_T}

    rows = []
    print(f"{'T':>3} | " + " | ".join(f"alpha={a:<3} geo-mean [95% CI] (cens)" for a in ALPHAS_FULL) + " | M*  MMDP cost")
    for T in all_T:
        cells = []
        for al in ALPHAS_FULL:
            r = cost.get((T, al))
            if r is None:
                cells.append(f"{'-':>34}"); continue
            v = [math.log10(x) for x in r if x is not None]
            if v:
                m, lo, hi = bootstrap_ci(v)
                rows.append(["alpha", T, al, 10 ** m, 10 ** lo, 10 ** hi, len(r) - len(v)])
                cells.append(f"{10**m:>9.0f} [{10**lo:>8.0f},{10**hi:>9.0f}] ({len(r)-len(v)})")
            else:
                cells.append(f"{'all censored':>34}")
        ms = mstar[T]
        if ms:
            rows.append(["mmdp", T, ms, ms * T * (T + 1) // 2, "", "", 0])
        print(f"{T:>3} | " + " | ".join(cells) + f" | {str(ms):>3} {str(ms * T * (T + 1) // 2 if ms else None):>9}")
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "T", "alpha_or_Mstar", "geo_mean_interactions", "ci_lo", "ci_hi", "censored"])
        w.writerows(rows)
    print(f"bound-based crossover (constants=1, delta={DELTA}): T = {bound_crossover()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for al, c in zip(ALPHAS_FULL, ["#1f4e9c", "#7a7a7a", "#c0392b"]):
        Ts = [T for T in all_T if any(x is not None for x in cost.get((T, al), [None]))]
        g = [bootstrap_ci([math.log10(x) for x in cost[(T, al)] if x is not None]) for T in Ts]
        ax[0].plot(Ts, [x[0] for x in g], "o-", color=c, label=rf"$\alpha={al}$")
        ax[0].fill_between(Ts, [x[1] for x in g], [x[2] for x in g], color=c, alpha=0.15)
    Tm = [T for T in all_T if mstar[T]]
    ax[0].plot(Tm, [math.log10(mstar[T] * T * (T + 1) / 2) for T in Tm], "s-", color="#2e8b57", label="MMDP (sampled, $M^*$)")
    Tr = np.arange(2, max(all_T) + 1)
    ax[0].plot(Tr, Tr * math.log10(A), "k:", label=r"lower bound $|A|^T$ (standard IRL only)")
    ax[0].plot(Tr, [math.log10(bound_total(int(T))) for T in Tr], "r--", label="MMDP upper bound (const.=1)")
    if max(all_T) >= bound_crossover():
        ax[0].axvline(bound_crossover(), color="gray", ls="--", lw=0.8)
    ax[0].set_xlabel("horizon $T$"); ax[0].set_ylabel(r"$\log_{10}$ interactions")
    ax[0].set_title(r"Tree: interactions to reach gap $\leq 0.5\,V_{max}$"); ax[0].legend(fontsize=8)
    ax[1].semilogy(Tm, [mstar[T] for T in Tm], "s-", color="#2e8b57", label="measured $M^*$")
    ax[1].semilogy(Tr, [bound_M(int(T)) for T in Tr], "r--", label="bound on $M$ (const.=1)")
    ax[1].set_xlabel("horizon $T$"); ax[1].set_ylabel("samples per step $M$"); ax[1].legend()
    ax[1].set_title("Samples per MMDP step")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "cost_vs_T.png"), dpi=150)
    print(f"saved raw_runs.csv, summary.csv, cost_vs_T.png in ./{args.out}/")


if __name__ == "__main__":
    main()
