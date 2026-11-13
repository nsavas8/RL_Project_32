"""
Tabular MMDP (Moment Matching by Dynamic Programming), Swamy et al. 2023, Alg. 3,
checked on the paper's toy MDPs (DANTE, Cliff).  Everything is exact (no sampling):
expert visitation, suffix visitation counts and the per-step game are computed in closed form.

Conventions
-----------
* Time t = 1..T, a policy pis[t] is an (nS, nA) row-stochastic array.
* Deterministic dynamics nxt[s, a]; reward r[s, a]; J(pi) = sum_t E[r(s_t, a_t)].
* Per-step game (paper Eq. 1/2, sign as in Eq. 2: expert minus learner):
      pi_t = argmin_pi  max_{f in F}  (1/T) * < f , mu_E - mu_pi >
  where mu_pi are expected (s,a) visit counts of the suffix tau >= t when the learner starts in
  s ~ rho_E^t, plays a ~ pi(s) and then follows the ALREADY LEARNED pi_{t+1:T};
  mu_E is the same but with the first action drawn from the expert.
* Reward classes F (this is where the interesting subtlety is):
    'box' : all f in [-1,1]^{S x A}      (closed under negation) -> max_f <f,d> = ||d||_1
    'sym' : F = {r, -r}                   (closed under negation) -> |<r,d>|
    'one' : F = {r}                       (NOT closed under negation) -> <r,d>
* Unsupported states (never visited by the expert at time t) get a default action.
* A tiny entropy bonus (MaxEnt-style) breaks ties.
"""
import numpy as np

CLASSES = ["box", "sym", "one"]
STAY, UP, DOWN = 0, 1, 2      # DANTE action order, identical to dante_port.py


# --------------------------------------------------------------------------- MDP
class MDP:
    def __init__(self, nxt, r, s0, T, default_a, name):
        self.nxt, self.r = np.asarray(nxt), np.asarray(r, float)
        self.nS, self.nA = self.nxt.shape
        self.K, self.s0, self.T, self.default_a, self.name = self.nS * self.nA, s0, T, default_a, name


def onehot_policy(mdp, a):
    pi = np.zeros((mdp.nS, mdp.nA)); pi[:, a] = 1.0
    return pi


def step_dist(mdp, dist, pi):
    new = np.zeros(mdp.nS)
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            new[mdp.nxt[s, a]] += dist[s] * pi[s, a]
    return new


def dante(T):
    """Same conventions as dante_port.py: rows 0=top,1=center,2=bottom; actions 0=stay, 1=up, 2=down;
    reward 1 when the NEXT state is not the bottom row. Tremble-free (exact computation)."""
    nxt = np.array([[s, max(s - 1, 0), min(s + 1, 2)] for s in range(3)])
    r = np.array([[float(nxt[s, a] != 2) for a in range(3)] for s in range(3)])
    return MDP(nxt, r, s0=1, T=T, default_a=0, name="DANTE")


def cliff(T):
    """Chain s_0..s_{T-1}, absorbing cliff state sx. a1 advances, a2 falls. r = -1[s=sx] - 1[a=a2]."""
    sx = T
    nxt = np.zeros((T + 1, 2), int)
    for s in range(T):
        nxt[s, 0] = min(s + 1, T - 1); nxt[s, 1] = sx
    nxt[sx] = sx
    r = np.zeros((T + 1, 2)); r[:, 1] -= 1.0; r[sx, :] -= 1.0
    return MDP(nxt, r, s0=0, T=T, default_a=0, name="Cliff")


# --------------------------------------------------------------------------- evaluation
def expert_rho(mdp, piE):
    rho, dist = [None] * (mdp.T + 2), np.zeros(mdp.nS); dist[mdp.s0] = 1
    for t in range(1, mdp.T + 1):
        rho[t] = dist; dist = step_dist(mdp, dist, piE)
    return rho


def value(mdp, pis):
    dist, tot = np.zeros(mdp.nS), 0.0; dist[mdp.s0] = 1
    for t in range(1, mdp.T + 1):
        tot += (dist[:, None] * pis[t] * mdp.r).sum(); dist = step_dist(mdp, dist, pis[t])
    return tot


def suffix_counts(mdp, pis, t):
    """C[s,a,:] = expected (s,a) visit counts over times t..T: play a in s at time t, then pis[t+1..T]."""
    C = np.zeros((mdp.nS, mdp.nA, mdp.K))
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            c = np.zeros((mdp.nS, mdp.nA)); c[s, a] = 1
            dist = np.zeros(mdp.nS); dist[mdp.nxt[s, a]] = 1
            for tau in range(t + 1, mdp.T + 1):
                c += dist[:, None] * pis[tau]; dist = step_dist(mdp, dist, pis[tau])
            C[s, a] = c.ravel()
    return C


def game_d(mdp, pis, t, rho_t, piE):
    C = suffix_counts(mdp, pis, t)
    return np.einsum('s,sa,sak->k', rho_t, piE, C) - np.einsum('s,sa,sak->k', rho_t, pis[t], C)


def objective(mdp, d, cls):
    T, r = mdp.T, mdp.r.ravel()
    return {"box": np.abs(d).sum() / T, "sym": abs(r @ d) / T, "one": (r @ d) / T}[cls]


def eps_list(mdp, pis, piE, cls):
    rho = expert_rho(mdp, piE)
    return [objective(mdp, game_d(mdp, pis, t, rho[t], piE), cls) for t in range(1, mdp.T + 1)]


# --------------------------------------------------------------------------- MMDP
def solve_step(mdp, pis, t, rho_t, piE, cls, lam=1e-5):
    import cvxpy as cp          # only needed here, so E1 and E2 run without cvxpy
    C = suffix_counts(mdp, pis, t)
    supp = [s for s in range(mdp.nS) if rho_t[s] > 1e-12]
    muE = np.einsum('s,sa,sak->k', rho_t, piE, C)
    p = cp.Variable((len(supp), mdp.nA), nonneg=True)
    muPi = sum(rho_t[s] * (p[i] @ C[s]) for i, s in enumerate(supp))
    d, r, T = muE - muPi, mdp.r.ravel(), mdp.T
    obj = {"box": cp.norm1(d) / T, "sym": cp.abs(r @ d) / T, "one": (r @ d) / T}[cls]
    ent = sum(rho_t[s] * cp.sum(cp.entr(p[i])) for i, s in enumerate(supp))
    cp.Problem(cp.Minimize(obj - lam * ent), [cp.sum(p, axis=1) == 1]).solve()
    pi = onehot_policy(mdp, mdp.default_a)
    for i, s in enumerate(supp):
        v = np.clip(p.value[i], 0, None); pi[s] = v / v.sum()
    return pi


def mmdp(mdp, piE, cls, perturb=None):
    """Backward pass t = T..1. `perturb` (optional) corrupts each solved pi_t, mimicking optimisation error."""
    rho, pis = expert_rho(mdp, piE), [None] * (mdp.T + 2)
    for t in range(mdp.T, 0, -1):
        pi = solve_step(mdp, pis, t, rho[t], piE, cls)
        pis[t] = perturb(pi) if perturb else pi
    return pis


def bc(mdp, piE, perturb=None):
    rho, pis = expert_rho(mdp, piE), [None] * (mdp.T + 2)
    for t in range(1, mdp.T + 1):
        pi = onehot_policy(mdp, mdp.default_a)
        for s in range(mdp.nS):
            if rho[t][s] > 1e-12: pi[s] = piE[s]
        pis[t] = perturb(pi) if perturb else pi
    return pis


# --------------------------------------------------------------------------- experiments
def exp1_dante_one_step(eps=0.01):
    print("\n=== E1: DANTE, choose pi_1 given a fixed faulty suffix (paper Sec. 3.2 example) ===")
    print("suffix: pi_3:T = stay, pi_2 = go down w.p. p = eps*T.  Paper: BC gap = p(T-1), MMDP gap = 0.")
    for T in [10, 20, 50]:
        m, p = dante(T), eps * T
        piE = onehot_policy(m, STAY)
        pis = [None] + [None] * (T + 1)
        for t in range(3, T + 1): pis[t] = onehot_policy(m, STAY)
        pis[2] = (1 - p) * onehot_policy(m, STAY) + p * onehot_policy(m, DOWN)
        rho = expert_rho(m, piE)
        print(f"\nT={T}, p={p:.2f}, paper formula p(T-1) = {p*(T-1):.3f}")
        print(f"{'pi_1 action':>12} | " + " | ".join(f"game[{c}]" for c in CLASSES) + " | true gap J(piE)-J(pi)")
        for a, name in enumerate(["stay(=BC)", "up", "down"]):
            pis[1] = onehot_policy(m, a)
            d = game_d(m, pis, 1, rho[1], piE)
            gap = value(m, [None] + [onehot_policy(m, STAY)] * T + [None]) - value(m, pis)
            print(f"{name:>12} | " + " | ".join(f"{objective(m, d, c):8.4f}" for c in CLASSES) + f" | {gap:8.3f}")


def exp2_cliff_tightness(eps=0.01):
    print("\n=== E2: Cliff MDP, tightness of Thm 3.3 / 3.4 (pi_1 errs w.p. p=eps*T, rest perfect) ===")
    print(f"{'T':>4} {'class':>5} {'eps_bar':>9} {'eps_bar*T^2':>12} {'true gap':>9} {'gap<=bound':>11} {'tight(ratio)':>13}")
    for T in [10, 20, 50]:
        m, p = cliff(T), eps * T
        piE = onehot_policy(m, 0)
        pis = [None] + [onehot_policy(m, 0) for _ in range(T)] + [None]
        pis[1] = (1 - p) * onehot_policy(m, 0) + p * onehot_policy(m, 1)
        gap = value(m, [None] + [piE] * T + [None]) - value(m, pis)
        for c in CLASSES:
            eb = np.mean(eps_list(m, pis, piE, c)); bound = eb * T ** 2
            print(f"{T:>4} {c:>5} {eb:9.4f} {bound:12.3f} {gap:9.3f} {str(gap <= bound + 1e-9):>11} {gap / bound:13.3f}")


def exp3_exact_mmdp():
    print("\n=== E3: full backward MMDP, exact game solves (no optimisation error) ===")
    for m in [dante(20), cliff(20)]:
        piE = onehot_policy(m, 0)
        JE = value(m, [None] + [piE] * m.T + [None])
        for c in CLASSES:
            pis = mmdp(m, piE, c)
            print(f"{m.name:>6} T={m.T} class={c:>3}:  J(piE)={JE:6.2f}  J(MMDP)={value(m, pis):6.2f}  "
                  f"eps_bar={np.mean(eps_list(m, pis, piE, c)):+.2e}")


def exp4_dante_horizon(q=0.01, Ts=(10, 20, 50, 100)):
    print(f"\n=== E4: DANTE, every learned pi_t is corrupted (goes down w.p. q={q}); gap vs horizon T ===")
    print("Models optimisation error eps_t. BC = per-step expert matching; MMDP sees the corrupted suffix.")
    print(f"{'T':>4} {'BC':>8} {'MMDP box':>9} {'MMDP sym':>9} {'MMDP one':>9}   q*T*(T-1)/2")
    for T in Ts:
        m = dante(T); piE = onehot_policy(m, STAY)
        perturb = lambda pi: (1 - q) * pi + q * onehot_policy(m, DOWN)
        JE = value(m, [None] + [piE] * T + [None])
        row = [JE - value(m, bc(m, piE, perturb))]
        for c in CLASSES: row.append(JE - value(m, mmdp(m, piE, c, perturb)))
        print(f"{T:>4} " + " ".join(f"{g:8.3f}" for g in row) + f"   {q*T*(T-1)/2:8.3f}")


if __name__ == "__main__":
    exp1_dante_one_step()
    exp2_cliff_tightness()
    exp3_exact_mmdp()
    exp4_dante_horizon()
