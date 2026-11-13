"""
Tabular FILTER learner (Swamy et al. 2023, Sec. 4 + Alg. 4).
 
One rollout (one "sample" in NRMM / NRPI):
    1. draw t ~ Unif{0..T-1}
    2. with prob alpha:   expert reset, s_t ~ rho_E^t            (free)
       otherwise:         start in s0 and roll in pi for t steps  (costs t interactions)
    3. take a uniformly random action a_t in s_t, then follow pi to the end (costs T - t)
    4. record Q_hat(s_t, a_t) = sum of rewards from t to T
 
Policy update (after every batch of M rollouts), FTRL / follow-the-leader on ALL
data so far:  pi(s) = argmax_a mean Q_hat(s, a), uniform over ties.
States never used as a start s_t stay uniformly random. This is the key point:
the policy only learns in states where rollouts *start*, so alpha decides
which states it learns in.
 
  alpha = 1  -> NRMM:  starts only on expert states -> cheap, but never learns to recover.
  alpha = 0  -> standard IRL (primal, Alg. 2 with an off-policy-style inner loop):
                starts on its own distribution -> learns recovery, but must explore from s0.
 
Simplification (same as the tree): the reward is fixed to the true reward
r = 1[next state = C] instead of being chosen by a learned discriminator.
"""

import numpy as np

class FilterLearner: 
    def __init__(self, env, alpha=0.5, rollouts_per_iter=10, seed=0):
        self.env = env
        self.alpha = alpha
        self.M = rollouts_per_iter
        self.rng = np.random.default_rng(seed)
        self.q_sum = np.zeros((env.nS, env.nA))
        self.q_cnt = np.zeros((env.nS, env.nA))
        self.pi = np.full((env.nS, env.nA), 1.0/env.nA)

    def update_policy(self):
        mean = np.divide(self.q_sum, self.q_cnt, out=np.zeros_like(self.q_sum), where=self.q_cnt > 0)
        best = mean == mean.max(axis=1, keepdims=True)
        self.pi = best/best.sum(axis=1, keepdims=True)

    def act(self, s):
        return self.rng.choice(self.env.nA, p=self.pi[s])

    def rollout(self, alpha):
        env, T = self.env, self.env.T
        t = int(self.rng.integers(T))
        expert_reset = self.rng.random() < alpha
        if expert_reset:
            s = env.reset_to(env.expert_state(t), t)
        else:
            s = env.reset()
            for _ in range(t):
                s, _, _ = env.step(self.act(s))
        s_t = s
        a_t = int(self.rng.integers(env.nA))
        s, q_hat, done = env.step(a_t)
        while not done: 
            s, r, done = env.step(self.act(s))
            q_hat += r
        self.q_sum[s_t, a_t] += q_hat / (T-t)
        self.q_cnt[s_t, a_t] += 1
        return s_t, expert_reset

    def run(self, budget):
        env = self.env
        J_E = env.evaluate(env.expert_policy())
        log = {"interactions": [], "gap": [], "alpha": [], "frac_offsupport_starts": []}
        support, n_off, n_total = env.expert_support(), 0, 0
        start = env.n_interactions
        while env.n_interactions - start < budget:
            frac = (env.n_interactions - start) / budget
            a = self.alpha(frac) if callable(self.alpha) else self.alpha
            for _ in range(self.M):
                s_t, _ = self.rollout(a)
                n_off += s_t not in support
                n_total += 1
            self.update_policy()
            log["interactions"].append(env.n_interactions - start)
            log["gap"].append(J_E - env.evaluate(self.pi))
            log["alpha"].append(a)
            log["frac_offsupport_starts"].append(n_off / n_total)
        return {k: np.array(v) for k, v in log.items()}