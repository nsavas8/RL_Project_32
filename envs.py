"""
Far-start environment for claim A2 (alpha trade-off).
 
Layout (k = start distance, u = cost of a mistake, |A| = n_actions):
 
    s0 = L0 -> L1 -> ... -> L_{k-1} -> C  (corridor, the expert stays here)
                                        |
                       slip (wrong a)   v
                                       OFF_u -> ... -> OFF_1 -> C   (action 1 climbs back up)
 
  * Lock L_j:   action 0 advances, every other action -> LOST (absorbing, no reward).
                No reward in the lock, so a learner starting at s0 gets no signal
                until it has guessed the right k actions: exploration hardness ~ |A|^k.
  * Corridor C: action 0 stays on C. Any other action slips to OFF_u.
  * OFF_d:      action 1 climbs to OFF_{d-1} (OFF_0 = C), any other action sinks
                to OFF_{min(d+1, u)}. A slip therefore costs >= u steps even if the
                learner knows how to recover, and much more if it does not.
  * Reward:     r = 1 whenever the next state is C. So Vmax = J(expert, no tremble) = T - k + 1.
  * Tremble:    with prob p the executed action is uniformly random (as p_tremble in
                Swamy et al., Sec. 5). This is what makes errors compound at test time.
 
The expert's demonstrations are tremble-free, so its visitation is
rho_E^t = L_t for t < k and C for t >= k. These are the expert-reset states.
"""

import numpy as np

class FarStartEnv:
    # stores params and numbers of the states
    def __init__(self, T=50, k=4, n_actions=3, u=3, p_tremble=0.05, seed=0):
        assert 0 <= k < T and u >= 1 and n_actions >=2
        self.T, self.k, self.nA, self.u, self.p = T, k, n_actions, u, p_tremble
        self.C = k
        self.OFF = {d: k + d for d in range(1, u+1)}
        self.LOST = k + u + 1
        self.nS = k + u + 2
        self.s0 = 0 if k > 0 else self.C
        self.rng = np.random.default_rng(seed)
        self.n_interactions = 0
        self.s, self.t = self.s0, 0

    # rules of the MDP: given a state and the action actually executed, return the next state
    def next_state(self, s, a):
        if s < self.k:
            if a != 0:
                return self.LOST
            return s + 1
        if s == self.C:
            return self.C if a == 0 else self.OFF[self.u]
        if s == self.LOST:
            return self.LOST
        d = s - self.k
        return (self.C if d == 1 else self.OFF[d-1]) if a == 1 else self.OFF[min(d + 1, self.u)]

    # 1 if you land on C, else 0
    def reward(self, s_next):
        return 1.0 if s_next == self.C else 0.0

    # puts you back to s0 and t=0 (normal start, doesn't count as interaction)
    def reset(self):
        self.s, self.t = self.s0, 0
        return self.s

    # jumps to any state at time t -> expert reset (doesn't count as interaction)
    def reset_to(self, s, t):
        self.s, self.t = s, t 
        return self.s

    # replace action by random action (tremble) with probability p
    def step(self, a):
        if self.rng.random() < self.p:
            a = self.rng.integers(self.nA)
        s_next = self.next_state(self.s, a)
        r = self.reward(s_next)
        self.s, self.t = s_next, self.t + 1
        self.n_interactions += 1
        return s_next, r, self.t >= self.T

    # expert's state at time t
    def expert_state(self, t):
        return t if t < self.k else self.C

    # action 0 in the lock and on C, action 1 in OFF states
    def expert_policy(self):
        pi = np.zeros((self.nS, self.nA))
        pi[:, 0] = 1.0
        for s in self.OFF.values():
            pi[s] = 0.0
            pi[s, 1] = 1.0
        return pi

    # set of states the expert visits
    def expert_support(self):
        return set(range(self.k)) | {self.C}

    # exact J(π), pushes a probability distribution over states forward T times and adds up the probability of being on C
    def evaluate(self, pi, p_tremble=None):
        p = self.p if p_tremble is None else p_tremble
        eff = (1-p) * pi + p / self.nA
        if not hasattr(self, "_P"):
            self._P = np.zeros((self.nS, self.nA, self.nS))
            for s in range(self.nS):
                for a in range(self.nA):
                    self._P[s, a, self.next_state(s, a)] = 1.0
        P = self._P
        d = np.zeros(self.nS)
        d[self.s0] = 1.0
        J = 0.0
        for _ in range(self.T):
            d = np.einsum("s,sa,san->n", d, eff, P)
            J += d[self.C]
        return J

    @property
    # J(expert) with no tremble = T-k+1
    def vmax(self):
        return self.evaluate(self.expert_policy(), p_tremble=0.0)