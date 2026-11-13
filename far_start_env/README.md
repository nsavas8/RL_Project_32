# A2 trade-off: expert-reset probability α 

Part of Project 32, *[Re] Inverse Reinforcement Learning without Reinforcement Learning* (Swamy et al., ICML 2023).

**Question (A2b):** how does the expert-reset probability α trade the number of environment interactions against the accumulation of errors, and which α is best for a given interaction budget?

Status (13 Nov 2026): environment and learner work, and a first smoke test (10 seeds) shows the expected trade-off. The full α sweep, the statistics and the plots are not done yet.

## Files

| File | Content |
|---|---|
| `envs.py` | `FarStartEnv`, the toy environment |
| `learner.py` | `FilterLearner`, a tabular FILTER learner with expert-reset probability α |
| `smoke_test.py` | Sanity checks plus a first α comparison (prints a table) |

Run with `python3 smoke_test.py` (about 3 min; needs only numpy).

## Environment: `FarStartEnv`

```
s0 = L0 → L1 → L2 → L3 → C   (corridor, reward 1 per step)
                         ↓ slip (any action ≠ 0, or tremble)
              OFF_3 → OFF_2 → OFF_1 → C   (action 1 climbs back up)
```

- **Lock L0…L(k−1):** action 0 advances; any other action leads to LOST (absorbing). No reward in the lock, so a learner starting at s0 must guess the whole sequence, about |A|^k tries. This is the tree of Thm 3.1 (paper A.2), with all wrong subtrees merged into LOST.
- **Corridor C:** reward 1 each time the agent lands on C; action 0 stays.
- **OFF states:** action 1 climbs one level, other actions sink. A slip costs at least u steps, and the rest of the episode if the learner never learned to climb.
- **Tremble:** with probability p the executed action is replaced by a random one (as `p_tremble` in paper Sec. 5). Without it, compounding errors never show up.
- **Expert:** action 0 in the lock and on C. Its tremble-free visitation is ρ_E^t = L_t for t < k and C for t ≥ k; expert resets can only jump there. Off its path (OFF states) the expert is defined by us as optimal (climb).
- **Vmax** = J(expert, no tremble) = T − k + 1.

| Parameter | Value | Reason |
|---|---|---|
| T | 50 | Agreed in the team; T² vs T growth clearly visible |
| k | 4 | \|A\|^k = 81: α=0 is clearly slower but still finishes |
| \|A\| | 3 | As in DANTE; with 2 actions random recovery is too easy |
| u | 3 | A random policy almost never recovers, so α=1 really is stuck |
| p | 0.05 | Within the paper's range (0.01 to 0.1); about 1 slip per episode |

Interaction accounting: only `step()` counts. Resets (`reset`, `reset_to`) and evaluation (`evaluate`, exact) are free, as in the paper, which plots J against interactions used for policy optimisation.

## Learner: `FilterLearner`

NRMM (paper Alg. 4) plus the α mixture of Sec. 4. One rollout:

1. Draw t ~ Unif{0…T−1}.
2. With probability α do an **expert reset** to ρ_E^t (free). Otherwise **start at s0** and roll in with π for t steps.
3. Take a uniformly random action a_t at s_t, then follow π until T.
4. Store Q̂(s_t, a_t) = (sum of rewards from t to T) / (T − t), **for the start state only.**

After every M = 10 rollouts: π(s) = argmax_a of mean Q̂(s, a) over all data so far (follow-the-leader, FTRL without a regulariser), with ties split evenly. States never used as a start state stay uniform.

**Key mechanism:** the learner only learns in states where rollouts start, so α decides where it learns. α=1 learns only on expert states (cheap, never learns to recover). α=0 also learns on its own states (learns recovery, but must get through the lock first).

### Deviations from the paper (state in the report)

1. **No learned discriminator:** the reward class is F_r = {r}, so the best-response discriminator is r itself (same choice as the tree and Cliff).
2. **Q̂ normalised by remaining steps T − t** (see "Run 2" below for why).
3. **Last policy reported** instead of "the policy with the lowest validation error".
4. Environment is our own construction (lock after paper A.2, corridor after DANTE Sec. 3.2 and Cliff A.9).

## Results so far

Mean gap J(π_E) − J(π) over 10 seeds (lower is better; J(π_E) with tremble = 37.27, Vmax = 47). Sanity line: `Vmax = 47.0, J(expert, tremble) = 37.27, J(uniform) = 0.041`.

### Run 1: raw Q̂ (no normalisation)

| α | 2k | 20k | 100k | 400k | off-support starts |
|---|---|---|---|---|---|
| 0 | 37.23 | 33.46 | 17.47 | 2.69 | 46.9% |
| 0.25 | 36.85 | 32.53 | 4.91 | 1.64 | 24.8% |
| 0.5 | 36.76 | 28.29 | 2.98 | 0.00 | 15.0% |
| 0.75 | 36.96 | 21.86 | 1.82 | 2.98 | 7.3% |
| 1 | 36.28 | 14.62 | 11.21 | 11.21 | 0.0% |
| 1→0 | 36.28 | 13.43 | 3.22 | 3.12 | 11.5% |

Problem: per seed, the runs end either at 0 or around 14 to 16, and one run (α=0.75, seed 6) was at 0 after 100k and got *worse* afterwards. Inspection showed the learner picking the wrong action in OFF states (e.g. sinking instead of climbing). Cause: one table Q̂(s, a) mixes samples from different start times t. A sample from early t has many remaining steps and therefore a large return. With only a handful of samples per action in OFF states, the action that happened to be sampled at earlier times wins the argmax.

### Run 2: Q̂ normalised by T − t (current code)

| α | 2k | 20k | 100k | 400k | seeds at gap 0 after 400k |
|---|---|---|---|---|---|
| 0 | 37.23 | 33.46 | 10.73 | 0.00 | 10/10 |
| 0.25 | 36.88 | 30.26 | 1.64 | 0.00 | 10/10 |
| 0.5 | 36.75 | 28.00 | 1.34 | 0.00 | 10/10 |
| 0.75 | 36.90 | 21.86 | **0.00** | 0.00 | 10/10 |
| 1 | 36.28 | 14.62 | 11.21 | 11.21 | 0/10 (all stuck at 11.21) |
| 1→0 | 36.28 | **12.36** | 1.64 | 0.00 | 10/10 |

### Reading the results

- **At 2k nobody has learned anything yet:** the gap ≈ J(π_E), since a random policy almost never gets through the lock.
- **Small budget (about 20k): α=1 and the annealed schedule are best.** Expert resets learn the lock without exploration.
- **α=1 stops improving at a gap of 11.21 on every seed.** It never starts in OFF states (0% off-support starts), so it never learns to climb, and every slip costs the rest of the episode. At T=30 the same learner was stuck at about 4.5; a factor 2.5 for a 1.7× longer corridor fits error growing roughly like T² (Thm 3.8).
- **Large budget: every α < 1 reaches the expert.** The best α moves down as the budget grows: at 100k, 0.75 is best; α=0 is still the slowest because of the lock.
- **The best α depends on the budget** (A2b): close to 1 for small budgets, anything below 1 for large ones, with 0.5 to 0.75 or annealing as a good compromise.

## Open points and next steps

- [ ] **Align with Jana:** her tree learner updates every visited state; this learner updates only the start state (as in the paper). With update-everywhere, α=1 would also learn to climb and the trade-off would mostly disappear. Decide on one learner, or report both.
- [ ] Full α sweep: α ∈ {0, 0.25, 0.5, 0.75, 1, 1→0}, **20 seeds**, 95% CIs; **save logs** (`np.savez`) instead of only printing.
- [ ] Metric for "cost to reach target", e.g. interactions until gap ≤ 0.5·Vmax (or a stricter target), plus the share of seeds that reach it.
- [ ] Plots: cost vs error per α (A2b, Fig. 8a) and best α vs budget (Fig. 8b).
- [ ] Stretch (A2c): vary k, |A|, u separately; run the same learner on the tree, DANTE and Cliff.
- [ ] Optional check: a learned discriminator restricted to corridor features, to show the ordering of α doesn't change.
- [ ] Ask TA: is a fixed reward (F_r = {r}) acceptable, or is a learned discriminator expected?

## Bugs fixed so far

- `expert_state(t)` had the comparison flipped (`t > k` instead of `t < k`), so expert resets landed in wrong states, and crashed for t ≥ 9. It's worth testing this function on its own: `[env.expert_state(t) for t in range(8)]` should give `[0, 1, 2, 3, 4, 4, 4, 4]`.
