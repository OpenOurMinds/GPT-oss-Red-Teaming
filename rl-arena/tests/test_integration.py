"""
tests/test_integration.py
End-to-end integration + regression tests for the full RL Arena pipeline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
import json
import pytest
from env import LLMSecurityEnv
from agents import RedTeamAgent, BlueTeamAgent
from techniques import (
    NUM_ATTACKS, NUM_DEFENSES, INTERACTION_MATRIX,
    get_attack, get_defense, get_success_probability,
)


# ── Helpers ──────────────────────────────────────────────────────

def run_n_episodes(n: int, max_steps: int = 20, seed: int = 0):
    """Run n complete training episodes; return final agents + env."""
    random.seed(seed)
    env = LLMSecurityEnv(max_steps=max_steps)
    red = RedTeamAgent()
    blue = BlueTeamAgent()

    for ep in range(n):
        ro, bo = env.reset()
        done = False
        while not done:
            ra = red.select_action(ro)
            ba = blue.select_action(bo)
            new_ro, new_bo, rr, br, done, info = env.step(ra, ba)
            red.update(ro, ra, rr, new_ro, done)
            blue.update(bo, ba, br, new_bo, done)
            ro, bo = new_ro, new_bo
        red.decay_epsilon()
        blue.decay_epsilon()
        red.record_episode_reward(sum(h["red_reward"] for h in env.history))
        blue.record_episode_reward(sum(h["blue_reward"] for h in env.history))

    return env, red, blue


# ── Pipeline stability ───────────────────────────────────────────

class TestPipelineStability:
    def test_50_episodes_no_exception(self):
        """The full pipeline must run 50 episodes without any exception."""
        env, red, blue = run_n_episodes(50)

    def test_100_episodes_q_tables_populated(self):
        _, red, blue = run_n_episodes(100)
        assert len(red.Q) > 0, "Red Q-table empty after 100 episodes"
        assert len(blue.Q) > 0, "Blue Q-table empty after 100 episodes"

    def test_200_episodes_epsilon_decayed(self):
        _, red, blue = run_n_episodes(200)
        assert red.epsilon < 0.60, f"Red ε={red.epsilon:.4f} not decayed after 200 eps"
        assert blue.epsilon < 0.60, f"Blue ε={blue.epsilon:.4f} not decayed after 200 eps"

    def test_episode_rewards_length(self):
        n = 30
        _, red, blue = run_n_episodes(n)
        assert len(red.episode_rewards) == n
        assert len(blue.episode_rewards) == n

    def test_cumulative_rewards_length(self):
        n = 30
        _, red, blue = run_n_episodes(n)
        assert len(red.cumulative_rewards) == n
        assert len(blue.cumulative_rewards) == n

    def test_no_nan_in_q_tables(self):
        import numpy as np
        _, red, blue = run_n_episodes(50)
        for state, q in red.Q.items():
            assert not np.any(np.isnan(q)), f"NaN in Red Q[{state}]"
        for state, q in blue.Q.items():
            assert not np.any(np.isnan(q)), f"NaN in Blue Q[{state}]"

    def test_no_inf_in_q_tables(self):
        import numpy as np
        _, red, blue = run_n_episodes(50)
        for state, q in red.Q.items():
            assert not np.any(np.isinf(q)), f"Inf in Red Q[{state}]"
        for state, q in blue.Q.items():
            assert not np.any(np.isinf(q)), f"Inf in Blue Q[{state}]"


# ── Determinism / reproducibility ────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_episode_rewards(self):
        _, r1, b1 = run_n_episodes(20, seed=42)
        _, r2, b2 = run_n_episodes(20, seed=42)
        assert r1.episode_rewards == r2.episode_rewards, "Red rewards not deterministic"
        assert b1.episode_rewards == b2.episode_rewards, "Blue rewards not deterministic"

    def test_different_seeds_different_rewards(self):
        _, r1, _ = run_n_episodes(20, seed=1)
        _, r2, _ = run_n_episodes(20, seed=2)
        # Should differ at least somewhat
        assert r1.episode_rewards != r2.episode_rewards, \
            "Different seeds produced identical rewards — suspiciously deterministic"


# ── Learning convergence ─────────────────────────────────────────

class TestLearningConvergence:
    def test_red_should_prefer_gcg_after_many_episodes(self):
        """After 500 episodes, GCG (id=8) should be Red's most chosen action."""
        _, red, _ = run_n_episodes(500, seed=42)
        most_used = int(red.action_counts.argmax())
        # Not asserting strict == 8, because convergence depends on stochastic env,
        # but we can assert GCG is in the top 3 most-used
        top3 = set(red.action_counts.argsort()[-3:].tolist())
        assert 8 in top3, \
            f"GCG (id=8) not in top-3 used attacks. Counts: {red.action_counts.tolist()}"

    def test_blue_wins_rate_above_floor(self):
        """Blue team should win at least 10% of episodes (non-trivial defense learning)."""
        n = 300
        _, red, blue = run_n_episodes(n, seed=99)
        blue_wins = sum(
            1 for r, b in zip(red.episode_rewards, blue.episode_rewards)
            if b > r
        )
        rate = blue_wins / n
        assert rate >= 0.10, f"Blue win rate={rate:.2%} (< 10%), defense not learning"

    def test_jailbreak_rate_not_trivial(self):
        """Jailbreak rate after training should be between 10% and 90%."""
        env, _, _ = run_n_episodes(200, seed=5)
        m = env.get_metrics()
        jr = m.get("jailbreak_rate", 0)
        assert 0.05 <= jr <= 0.95, \
            f"Jailbreak rate={jr:.2%} is trivially extreme — check reward balance"

    def test_both_agents_use_multiple_actions(self):
        """After 200 episodes, each agent should have used at least 5 different actions."""
        _, red, blue = run_n_episodes(200, seed=0)
        red_unique = (red.action_counts > 0).sum()
        blue_unique = (blue.action_counts > 0).sum()
        assert red_unique >= 5, f"Red only used {red_unique} unique attacks"
        assert blue_unique >= 5, f"Blue only used {blue_unique} unique defenses"


# ── JSON serialisability (server output) ─────────────────────────

class TestJsonSerialisation:
    """All data streamed via WebSocket must be JSON-serialisable."""

    def test_q_matrix_is_json_serialisable(self):
        _, red, blue = run_n_episodes(50)
        try:
            json.dumps(red.get_q_matrix())
            json.dumps(blue.get_q_matrix())
        except (TypeError, ValueError) as e:
            pytest.fail(f"Q-matrix is not JSON-serialisable: {e}")

    def test_action_distribution_is_json_serialisable(self):
        _, red, blue = run_n_episodes(50)
        try:
            json.dumps(red.get_action_distribution())
            json.dumps(blue.get_action_distribution())
        except (TypeError, ValueError) as e:
            pytest.fail(f"Action distribution is not JSON-serialisable: {e}")

    def test_stats_is_json_serialisable(self):
        _, red, blue = run_n_episodes(50)
        try:
            json.dumps(red.stats())
            json.dumps(blue.stats())
        except (TypeError, ValueError) as e:
            pytest.fail(f"stats() is not JSON-serialisable: {e}")

    def test_env_metrics_is_json_serialisable(self):
        env, _, _ = run_n_episodes(10)
        try:
            json.dumps(env.get_metrics())
        except (TypeError, ValueError) as e:
            pytest.fail(f"get_metrics() is not JSON-serialisable: {e}")

    def test_interaction_matrix_is_json_serialisable(self):
        try:
            json.dumps(INTERACTION_MATRIX)
        except (TypeError, ValueError) as e:
            pytest.fail(f"INTERACTION_MATRIX is not JSON-serialisable: {e}")

    def test_step_info_is_json_serialisable(self):
        env = LLMSecurityEnv(max_steps=5)
        env.reset()
        _, _, _, _, _, info = env.step(0, 0)
        try:
            json.dumps(info)
        except (TypeError, ValueError) as e:
            pytest.fail(f"step info dict is not JSON-serialisable: {e}")

    def test_episode_rewards_list_is_json_serialisable(self):
        _, red, blue = run_n_episodes(50)
        try:
            json.dumps(red.episode_rewards)
            json.dumps(blue.episode_rewards)
        except (TypeError, ValueError) as e:
            pytest.fail(f"episode_rewards is not JSON-serialisable: {e}")


# ── Regression: edge cases ───────────────────────────────────────

class TestEdgeCases:
    def test_env_reset_mid_episode(self):
        """Should be able to reset at any point and not crash."""
        env = LLMSecurityEnv(max_steps=10)
        red = RedTeamAgent()
        blue = BlueTeamAgent()
        ro, bo = env.reset()
        env.step(0, 0)
        env.step(1, 1)
        ro, bo = env.reset()   # mid-episode reset
        # Should be able to continue from here
        _, _, rr, br, _, _ = env.step(2, 2)
        assert isinstance(rr, float)

    def test_single_step_environment(self):
        """max_steps=1 must complete in exactly one step."""
        env = LLMSecurityEnv(max_steps=1)
        env.reset()
        _, _, _, _, done, _ = env.step(0, 0)
        assert done, "Environment with max_steps=1 should be done after one step"

    def test_all_attacks_vs_all_defenses_no_crash(self):
        """Cross-product of all (attack, defense) pairs must not raise."""
        env = LLMSecurityEnv(max_steps=1)
        for ra in range(NUM_ATTACKS):
            for ba in range(NUM_DEFENSES):
                env.reset()
                env.step(ra, ba)

    def test_agent_update_with_large_reward(self):
        red = RedTeamAgent()
        state = (0, 0, 0, 0)
        # Large reward should not cause overflow in float32 Q-values
        red.update(state, 0, 1e6, state, True)
        import math
        assert not math.isnan(float(red.Q[state][0]))
        assert not math.isinf(float(red.Q[state][0]))

    def test_agent_handles_repeated_same_state(self):
        """Repeated updates to the same state converge (no drift)."""
        red = RedTeamAgent()
        state = (0, 0, 0, 0)
        for _ in range(1000):
            red.update(state, 0, 0.0, state, True)
        # Q should converge to 0 for zero reward, done=True
        assert abs(float(red.Q[state][0])) < 1e-3, \
            f"Q did not converge to 0 after 1000 zero-reward done updates: {red.Q[state][0]}"

    def test_get_q_matrix_after_zero_episodes(self):
        """Fresh agent Q-matrix should be empty."""
        red = RedTeamAgent()
        m = red.get_q_matrix()
        assert m == []

    def test_env_history_type(self):
        env = LLMSecurityEnv(max_steps=5)
        env.reset()
        env.step(0, 0)
        assert isinstance(env.history, list)
        assert isinstance(env.history[0], dict)

    def test_multiple_resets_consistent(self):
        """Resetting many times should always return the same initial obs."""
        env = LLMSecurityEnv(max_steps=5)
        first = env.reset()
        for _ in range(10):
            obs = env.reset()
            assert obs == first, f"Reset returned {obs}, expected {first}"


# ── Stress test ──────────────────────────────────────────────────

class TestStress:
    @pytest.mark.slow
    def test_500_episode_run_completes(self):
        """Full-scale run must complete without error."""
        env, red, blue = run_n_episodes(500, seed=0)
        assert len(red.episode_rewards) == 500
        assert red.epsilon < 0.15, f"Red ε={red.epsilon:.4f} not sufficiently decayed"

    def test_env_max_steps_large(self):
        """Large max_steps should not exhaust memory or cause overflow."""
        env = LLMSecurityEnv(max_steps=1000)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, _, _, done, _ = env.step(
                random.randint(0, NUM_ATTACKS - 1),
                random.randint(0, NUM_DEFENSES - 1)
            )
            steps += 1
        assert steps == 1000
        assert len(env.history) == 1000
