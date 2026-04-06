"""
agents.py — Tabular Q-Learning agents for Red Team and Blue Team.

Both agents use ε-greedy exploration with exponential decay.
Q-tables are dictionaries: state → [Q(s, a0), Q(s, a1), ...]
Bellman update: Q(s,a) ← Q(s,a) + α · [r + γ·max_a'Q(s',a') - Q(s,a)]
"""

import random
import numpy as np
from collections import defaultdict
from techniques import NUM_ATTACKS, NUM_DEFENSES


class QLearningAgent:
    """Generic tabular Q-Learning agent."""

    def __init__(
        self,
        n_actions: int,
        name: str,
        alpha: float = 0.15,      # Learning rate
        gamma: float = 0.90,      # Discount factor
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions     = n_actions
        self.name          = name
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: default all zeros
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

        # Tracking
        self.episode_rewards: list[float] = []
        self.cumulative_rewards: list[float] = []
        self._cum = 0.0
        self.action_counts = np.zeros(n_actions, dtype=int)

    # ── Core ────────────────────────────────────────────────

    def select_action(self, state) -> int:
        """ε-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        qs = self.Q[state]
        # Break ties randomly
        max_q = np.max(qs)
        best  = np.where(qs == max_q)[0]
        return int(random.choice(best))

    def update(self, state, action: int, reward: float, next_state, done: bool):
        """Bellman update."""
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        self.action_counts[action] += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_episode_reward(self, reward: float):
        self.episode_rewards.append(round(reward, 4))
        self._cum += reward
        self.cumulative_rewards.append(round(self._cum, 4))

    # ── Diagnostics ──────────────────────────────────────────

    def get_policy(self) -> list[int]:
        """Returns greedy action for each unique visited state (by index)."""
        return {str(k): int(np.argmax(v)) for k, v in list(self.Q.items())[:20]}

    def get_q_table_flat(self) -> list[list[float]]:
        """Returns all Q-values flattened for heatmap: list of [action_values]."""
        # Aggregate Q-values across states by mean per action
        if not self.Q:
            return [[0.0] * self.n_actions]
        rows = list(self.Q.values())
        return [list(np.mean(rows, axis=0))]

    def get_action_distribution(self) -> list[float]:
        """Softmax probability distribution over actions based on total Q-value."""
        total = self.action_counts.sum()
        if total == 0:
            return [1.0 / self.n_actions] * self.n_actions
        return [round(float(c / total), 4) for c in self.action_counts]

    def get_q_matrix(self) -> list[list[float]]:
        """
        Returns per-state Q-values for heatmap.
        Aggregates by state's last_action bucket.
        """
        # Collect Q-rows grouped by first dimension of state key (opponent's last action)
        buckets = defaultdict(list)
        for state, q_vals in self.Q.items():
            bucket_key = state[0]  # first element of state tuple
            buckets[bucket_key].append(q_vals)

        result = []
        for key in sorted(buckets.keys()):
            row = np.mean(buckets[key], axis=0).tolist()
            result.append([round(v, 3) for v in row])
        return result

    def stats(self) -> dict:
        return {
            "name":           self.name,
            "epsilon":        round(self.epsilon, 4),
            "states_visited": len(self.Q),
            "total_updates":  int(self.action_counts.sum()),
            "action_dist":    self.get_action_distribution(),
        }


# ── Specialised agents ─────────────────────────────────────

class RedTeamAgent(QLearningAgent):
    """Red Team (attacker) RL agent."""
    def __init__(self, **kwargs):
        super().__init__(n_actions=NUM_ATTACKS, name="Red Team", **kwargs)


class BlueTeamAgent(QLearningAgent):
    """Blue Team (defender) RL agent."""
    def __init__(self, **kwargs):
        super().__init__(n_actions=NUM_DEFENSES, name="Blue Team",
                         alpha=0.12, gamma=0.92,  # Slightly more conservative
                         **kwargs)
