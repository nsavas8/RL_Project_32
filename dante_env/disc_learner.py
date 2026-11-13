"""
Learner B (robustness check): Nina's FilterLearner with a LEARNED discriminator instead of the true reward.
Works with any env that has the same interface (reset, reset_to, step, expert_state, expert_policy, n_interactions,
nS, nA, T, evaluate). Learner A (learner.py) is not touched.

Differences to learner A (everything else is inherited):
  * reward  : r_hat(s,a) = f_bar(s,a), a table in [-1,1]  (class = all tables, closed under negation as in the paper).
              At the start f_bar = 0.
  * f update: after every batch of M rollouts, the discriminator best-responds to the current learner:
                f_new(s,a) = sign( mu_E(s,a) - mu_pi(s,a) )
              mu_E  = (s,a) frequencies of N_E expert demonstrations (generated ONCE, with the environment's tremble,
                      free: not counted as interactions),
              mu_pi = (s,a) frequencies of the learner's rollouts in this batch.
              f_bar = running mean of all f_new  (follow-the-leader on the discriminator side, as in FILTER).
  * the noise from the finite number N_E of demonstrations and from the finite batch is what produces per-step error.
Settings you may change: N_E (n_demos), rollouts_per_iter M.
"""
import numpy as np
from learner import FilterLearner


class DiscFilterLearner(FilterLearner):
    def __init__(self, env, alpha=0.5, rollouts_per_iter=10, seed=0, n_demos=20):
        super().__init__(env, alpha, rollouts_per_iter, seed)
        self.f_sum = np.zeros((env.nS, env.nA)); self.n_f = 0
        self.f_bar = np.zeros((env.nS, env.nA))
        self.batch = np.zeros((env.nS, env.nA))
        self.mu_E = self._expert_freq(n_demos)

    def _expert_freq(self, n_demos):
        env, n0 = self.env, self.env.n_interactions
        pi_E, cnt = env.expert_policy(), np.zeros((env.nS, env.nA))
        for _ in range(n_demos):
            s = env.reset()
            done = False
            while not done:
                a = int(self.rng.choice(env.nA, p=pi_E[s]))
                cnt[s, a] += 1
                s, _, done = env.step(a)
        env.n_interactions = n0            # demonstrations are free
        env.reset()
        return cnt / cnt.sum()

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
        self.batch[s, a_t] += 1
        q_hat = self.f_bar[s, a_t]
        s, _, done = env.step(a_t)
        while not done:
            a = self.act(s)
            self.batch[s, a] += 1
            q_hat += self.f_bar[s, a]
            s, _, done = env.step(a)
        self.q_sum[s_t, a_t] += q_hat / (T - t)
        self.q_cnt[s_t, a_t] += 1
        return s_t, expert_reset

    def update_policy(self):
        mu_pi = self.batch / max(self.batch.sum(), 1)
        self.f_sum += np.sign(self.mu_E - mu_pi); self.n_f += 1
        self.f_bar = self.f_sum / self.n_f
        self.batch[:] = 0
        super().update_policy()
