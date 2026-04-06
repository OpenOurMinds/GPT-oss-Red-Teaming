"""
env.py — LLMSecurityEnv: Custom Gym-style Markov Decision Process
for Red Team vs Blue Team adversarial reinforcement learning.

State space  (Red observer) : [last_blue_action, last_outcome, consecutive_failures, episode_step]
State space  (Blue observer): [last_red_action,  last_outcome, successful_blocks,    episode_step]
Action space (Red)   : {0..N_ATTACKS-1}
Action space (Blue)  : {0..N_DEFENSES-1}
"""

import random
import numpy as np
from techniques import (
    NUM_ATTACKS, NUM_DEFENSES,
    get_success_probability, get_attack, get_defense
)


class LLMSecurityEnv:
    """
    Turn-based adversarial MDP: Red attacks, Blue defends.
    One 'step' = Red picks a technique → Blue picks a defense → outcome resolved.
    """

    # Outcome codes
    OUTCOME_JAILBREAK   = 2   # Attack bypassed defense
    OUTCOME_PARTIAL     = 1   # Partial compliance (some leakage)
    OUTCOME_BLOCKED     = 0   # Defense succeeded

    # Reward config
    RED_REWARD = {
        OUTCOME_JAILBREAK: +1.5,
        OUTCOME_PARTIAL:   +0.3,
        OUTCOME_BLOCKED:   -0.6,
    }
    BLUE_REWARD = {
        OUTCOME_JAILBREAK: -1.5,
        OUTCOME_PARTIAL:   -0.3,
        OUTCOME_BLOCKED:   +1.0,
    }

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self.reset()

    # ── Public API ──────────────────────────────────────────

    def reset(self):
        """Reset environment to initial state."""
        self.step_count          = 0
        self.red_consecutive_fail = 0
        self.blue_streak          = 0
        self.total_red_reward     = 0.0
        self.total_blue_reward    = 0.0
        self.history              = []

        self.last_red_action  = 0
        self.last_blue_action = 0
        self.last_outcome     = self.OUTCOME_BLOCKED

        return self._red_obs(), self._blue_obs()

    def step(self, red_action: int, blue_action: int):
        """
        Resolve one turn.
        Returns: red_obs, blue_obs, red_reward, blue_reward, done, info
        """
        assert 0 <= red_action  < NUM_ATTACKS,  f"Invalid red action {red_action}"
        assert 0 <= blue_action < NUM_DEFENSES, f"Invalid blue action {blue_action}"

        # ── Resolve outcome ─────────────────────────────────
        p_success = get_success_probability(red_action, blue_action)
        # Add stochastic curriculum noise (softens early episodes)
        noise = random.gauss(0, 0.05)
        p_eff = np.clip(p_success + noise, 0.02, 0.98)

        roll = random.random()
        if roll < p_eff * 0.85:
            outcome = self.OUTCOME_JAILBREAK
        elif roll < p_eff:
            outcome = self.OUTCOME_PARTIAL
        else:
            outcome = self.OUTCOME_BLOCKED

        # ── Compute rewards ─────────────────────────────────
        red_reward  = self.RED_REWARD[outcome]
        blue_reward = self.BLUE_REWARD[outcome]

        # Combo bonuses
        if outcome == self.OUTCOME_BLOCKED:
            self.blue_streak += 1
            self.red_consecutive_fail += 1
            if self.blue_streak >= 3:
                blue_reward += 0.4   # Blue streak bonus
        else:
            self.blue_streak = 0
            self.red_consecutive_fail = 0

        # Exploration bonus for red using novel techniques
        used_attacks = [h["red_action"] for h in self.history]
        if red_action not in used_attacks:
            red_reward += 0.1   # Novelty bonus

        # ── Update state ─────────────────────────────────────
        self.last_red_action  = red_action
        self.last_blue_action = blue_action
        self.last_outcome     = outcome
        self.step_count      += 1
        self.total_red_reward  += red_reward
        self.total_blue_reward += blue_reward

        attack  = get_attack(red_action)
        defense = get_defense(blue_action)

        info = {
            "step":           self.step_count,
            "red_action":     red_action,
            "blue_action":    blue_action,
            "attack_name":    attack.name,
            "attack_short":   attack.short,
            "defense_name":   defense.name,
            "defense_short":  defense.short,
            "outcome":        outcome,
            "outcome_label":  ["BLOCKED", "PARTIAL", "JAILBREAK"][outcome],
            "p_success":      round(p_eff, 3),
            "red_reward":     round(red_reward, 3),
            "blue_reward":    round(blue_reward, 3),
            "blue_streak":    self.blue_streak,
            "red_fail_streak":self.red_consecutive_fail,
            "attack_owasp":   attack.owasp,
            "defense_desc":   defense.description,
        }
        self.history.append(info)

        done = self.step_count >= self.max_steps
        return self._red_obs(), self._blue_obs(), red_reward, blue_reward, done, info

    def get_metrics(self):
        if not self.history:
            return {}
        outcomes = [h["outcome"] for h in self.history]
        jailbreaks = outcomes.count(self.OUTCOME_JAILBREAK)
        partials   = outcomes.count(self.OUTCOME_PARTIAL)
        blocks     = outcomes.count(self.OUTCOME_BLOCKED)
        total      = len(outcomes)
        return {
            "jailbreak_rate": round(jailbreaks / total, 3),
            "partial_rate":   round(partials   / total, 3),
            "block_rate":     round(blocks     / total, 3),
            "total_steps":    total,
            "red_score":      round(self.total_red_reward, 3),
            "blue_score":     round(self.total_blue_reward, 3),
        }

    # ── Private helpers ──────────────────────────────────────

    def _red_obs(self):
        """Red agent observes: [last_blue_action, last_outcome, fail_streak, step_norm]"""
        return (
            self.last_blue_action,
            self.last_outcome,
            min(self.red_consecutive_fail, 5),
            min(self.step_count, self.max_steps),
        )

    def _blue_obs(self):
        """Blue agent observes: [last_red_action, last_outcome, blue_streak, step_norm]"""
        return (
            self.last_red_action,
            self.last_outcome,
            min(self.blue_streak, 5),
            min(self.step_count, self.max_steps),
        )
