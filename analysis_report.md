# Deep Analysis: Tabular RL in MiniGrid (HW2)

## 1. Methodology & Observations

### Current Algorithm Performance (KeyDoorLavaEnv)
*   **SARSA (Success):** Reaches optimal solutions (~20 steps, 100% success). This is expected because SARSA is **on-policy**. In an environment with a "cliff" (Lava), SARSA learns the value of the policy it is actually following, including its exploration. This makes it more "cautious" and likely to avoid lava during training, leading to stable convergence.
*   **Monte Carlo (Sub-optimal):** Plateauing at ~100 steps. MC updates only happen at the end of an episode. If the agent reaches the goal via a very long, circuitous path, the entire path is reinforced. Without high-quality exploration, MC can get "trapped" in sub-optimal but successful loops.
*   **Q-Learning (Failure):** Stuck at -100 reward, 500 steps, 0% success. Q-Learning is **off-policy**, meaning it updates based on the *best* possible future action ($max Q$), regardless of the agent's actual next move. In environments with heavy penalties (Lava), Q-Learning's aggressive optimism can lead it to "suicide" into lava repeatedly during exploration, never seeing enough successful episodes to propagate the goal reward back.

## 2. Reward Shaping Analysis

### Current Implementation
*   **Step Penalty:** -0.2 (High, encourages speed).
*   **Goal Reward:** +100.0 (Strong signal).
*   **Event 1 (Failure):** -50.0 for Lava/Timeout.
*   **Event 2 (Door):** +30.0 for opening the door.

### Potential Issues
*   **The "Stay in Place" Stalling:** Even with -0.2 penalty, if the agent "thinks" all paths lead to lava (-50), it might prefer to wait and time out.
*   **The "Key" Gap:** We have a bonus for the *door*, but the *key* is a prerequisite. If the agent never finds the key, it never sees the door bonus.

### Proposed Improvement
*   **Add Key Bonus:** Trade the large Failure penalty for a small **Key Pickup bonus** (+10). This creates a smoother "gradient" of milestones: `Start -> Key (+10) -> Door (+30) -> Goal (+100)`.
*   **Scale Penalties:** Ensure the Lava penalty is large enough to be feared, but not so large that exploration becomes paralyzed.

## 3. Q-Table Initialization Strategy

### Current Strategy
*   Default (Zeros).

### Proposed Improvements
1.  **Optimistic Initialization:** Initialize all $Q(s,a)$ to a small positive constant (e.g., +1.0). This forces the agent to explore every action in every state at least once, as "unvisited" states will look better than "visited" states that have been hit by step penalties.
2.  **Manhattan Distance (Refined):** We previously tried absolute distance, but it failed due to randomized layouts. 
    *   *Correction:* Initialize based on **Relative Distance** `-(abs(dx) + abs(dy))` using the milestone-aware `dx, dy` from the state. This gives the agent a "compass" even before training starts.

## 4. Hyperparameter Analysis

*   **Gamma ($\gamma$):** 0.99 is correct for long horizons.
*   **Epsilon ($\epsilon$):** For randomized environments, a very high starting $\epsilon$ (1.0) is necessary. However, if using **Optimistic Initialization**, we can start with a *lower* $\epsilon$ (0.1) because the Q-table itself will drive exploration.
*   **Alpha ($\alpha$):** 0.1 is standard, but for the "moving target" of milestones, a slightly higher $\alpha$ (0.2) for Q-learning might help it lock onto the success signal faster.

## 5. Summary of Recommended Actions
1.  Add a bonus for picking up the key.
2.  Implement **Optimistic Initialization** to fix Q-learning's exploration paralysis.
3.  Increase training to 15,000 episodes for the most complex scenarios.
4.  Ensure `dx, dy` are properly bounded in the state to prevent an exploding Q-table size.
