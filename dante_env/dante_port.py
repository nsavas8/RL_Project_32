"""
DANTE in the interface of the far-start code (envs.py / learner.py), plus a sampled MMDP.
Her FilterLearner runs on DanteEnv unchanged (including its own run(budget) with logging).

DANTE (Swamy et al. 2023, Fig. 2 / Sec. 3.2): three rows, actions stay / up / down.
  states   0 = top, 1 = center, 2 = bottom;  s0 = center
  actions  0 = stay, 1 = up, 2 = down          (moves that would leave the grid keep the row)
  reward   1 whenever the NEXT state is not the bottom row   (goal: stay in the top two rows) -> Vmax = T
  tremble  with prob p the executed action is uniformly random (as p_tremble in Sec. 5 of the paper)
  expert   demonstrations (tremble-free) only ever visit the center row:  rho_E^t = {center}.
           expert_policy() is the optimal policy used for J(pi_E): stay in center/top, UP from the bottom row.
           (The demos do not show the recovery move; it is defined only so that the gap is to an achievable value.
            Same convention as expert_policy() in the far-start environment.)
"""
import math
import random

import numpy as np

TOP, CENTER, BOTTOM = 0, 1, 2
STAY, UP, DOWN = 0, 1, 2


class DanteEnv:
    def __init__(self, T=50, p_tremble=0.05, seed=0):
        self.T, self.p, self.nA, self.nS, self.s0 = T, p_tremble, 3, 3, CENTER
        self.rng = np.random.default_rng(seed)
        self.n_interactions = 0
        self.s, self.t = self.s0, 0

    def next_state(self, s, a):
        return s if a == STAY else (max(s - 1, 0) if a == UP else min(s + 1, 2))

    def reward(self, s_next):
        return 0.0 if s_next == BOTTOM else 1.0

    def reset(self):
        self.s, self.t = self.s0, 0
        return self.s

    def reset_to(self, s, t):                      # free (expert reset)
        self.s, self.t = s, t
        return self.s

    def step(self, a):                             # the only place where interactions are counted
        if self.rng.random() < self.p:
            a = int(self.rng.integers(self.nA))
        s_next = self.next_state(self.s, a)
        r = self.reward(s_next)
        self.s, self.t = s_next, self.t + 1
        self.n_interactions += 1
        return s_next, r, self.t >= self.T

    def expert_state(self, t):
        return CENTER

    def expert_support(self):
        return {CENTER}

    def expert_policy(self):
        pi = np.zeros((3, 3))
        pi[TOP, STAY] = pi[CENTER, STAY] = 1.0
        pi[BOTTOM, UP] = 1.0
        return pi

    def _P(self):
        if not hasattr(self, "_Pm"):
            self._Pm = np.zeros((3, 3, 3))
            for s in range(3):
                for a in range(3):
                    self._Pm[s, a, self.next_state(s, a)] = 1.0
        return self._Pm

    def evaluate(self, pi, p_tremble=None):
        """Exact J(pi) for a stationary policy pi (3x3). Free."""
        return self.evaluate_timed(lambda t: pi, p_tremble)

    def evaluate_timed(self, pi_of_t, p_tremble=None):
        """Exact J for a time-dependent policy: pi_of_t(t) -> 3x3 array, t = 0..T-1."""
        p = self.p if p_tremble is None else p_tremble
        P, d, J = self._P(), np.zeros(3), 0.0
        d[self.s0] = 1.0
        for t in range(self.T):
            eff = (1 - p) * pi_of_t(t) + p / 3
            d = np.einsum("s,sa,san->n", d, eff, P)
            J += d[CENTER] + d[TOP]
        return J

    @property
    def vmax(self):
        return self.evaluate(self.expert_policy(), p_tremble=0.0)


# ============================== sampled MMDP (Alg. 3) on DANTE ==============================
def run_mmdp(T, M, cls, seed, p_tremble=0.05):
    """For t = T-1..0: M samples = reset to the expert state (center) at time t (free), uniformly random first action,
    then the already learned pi_{t+1:T-1}. Policies are only learned on the expert's support (center); everywhere else
    the learner acts uniformly at random (as in Alg. 3). Importance weight |A| for the uniform first action.
    Reward class of the moment-matching game:
      'sym' : F = {r, -r} (closed under negation, the paper's assumption)  -> pi_t = argmin_a |Qhat(expert action) - Qhat(a)|
      'one' : F = {r}     (not closed under negation, as in Cliff)         -> pi_t = argmax_a Qhat(a)   (= PSDP)
    Returns (interactions, gap = J(pi_E) - J(pi)). Cost is exactly M*T*(T+1)/2 interactions."""
    env, rng = DanteEnv(T, p_tremble, seed), random.Random(f"dmmdp-{T}-{M}-{cls}-{seed}")
    pis = {}
    for t in range(T - 1, -1, -1):
        ret = [0.0] * 3
        for _ in range(M):
            a0 = rng.randrange(3)
            env.reset_to(CENTER, t)
            s, R, done = env.step(a0)
            while not done:
                a = rng.choices(range(3), weights=pis[env.t])[0] if (s == CENTER and env.t in pis) else rng.randrange(3)
                s, r, done = env.step(a)
                R += r
            ret[a0] += R
        q = [3 * ret[a] / M for a in range(3)]
        score = [-abs(q[STAY] - q[a]) for a in range(3)] if cls == "sym" else q
        best = [a for a in range(3) if score[a] >= max(score) - 1e-12]
        pis[t] = [1.0 / len(best) if a in best else 0.0 for a in range(3)]

    def pi_of_t(t):
        pi = np.full((3, 3), 1.0 / 3)
        pi[CENTER] = pis[t]
        return pi

    J_E = env.evaluate(env.expert_policy())
    return env.n_interactions, J_E - env.evaluate_timed(pi_of_t)
