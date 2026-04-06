"""
tests/test_env.py
Unit tests for env.py — LLMSecurityEnv MDP.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
import pytest
from env import LLMSecurityEnv
from techniques import NUM_ATTACKS, NUM_DEFENSES


@pytest.fixture
def env():
    """Fresh environment with fixed seed for reproducibility."""
    random.seed(0)
    return LLMSecurityEnv(max_steps=10)


@pytest.fixture
def env20():
    random.seed(42)
    return LLMSecurityEnv(max_steps=20)


# ── Reset ───────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_tuple_pair(self, env):
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_reset_obs_are_tuples(self, env):
        ro, bo = env.reset()
        assert isinstance(ro, tuple), f"red_obs is {type(ro)}, should be tuple"
        assert isinstance(bo, tuple), f"blue_obs is {type(bo)}, should be tuple"

    def test_reset_obs_length_four(self, env):
        ro, bo = env.reset()
        assert len(ro) == 4, f"red_obs length={len(ro)}, expected 4"
        assert len(bo) == 4, f"blue_obs length={len(bo)}, expected 4"

    def test_reset_step_count_zero(self, env):
        env.reset()
        assert env.step_count == 0

    def test_reset_clears_history(self, env20):
        ro, bo = env20.reset()
        ra, ba = 0, 0
        for _ in range(5):
            ro, bo, _, _, done, _ = env20.step(ra, ba)
            if done:
                break
        env20.reset()
        assert env20.history == []

    def test_reset_clears_rewards(self, env20):
        ro, bo = env20.reset()
        env20.step(0, 0)
        env20.reset()
        assert env20.total_red_reward == 0.0
        assert env20.total_blue_reward == 0.0

    def test_reset_clears_streaks(self, env20):
        env20.reset()
        env20.reset()  # double reset is safe
        assert env20.blue_streak == 0
        assert env20.red_consecutive_fail == 0

    def test_reset_initial_obs_values(self, env):
        ro, bo = env.reset()
        # last_blue_action / last_red_action initial = 0
        assert ro[0] == 0
        assert bo[0] == 0
        # last_outcome initial = BLOCKED (0)
        assert ro[1] == LLMSecurityEnv.OUTCOME_BLOCKED
        assert bo[1] == LLMSecurityEnv.OUTCOME_BLOCKED
        # step = 0
        assert ro[3] == 0
        assert bo[3] == 0

    def test_double_reset_idempotent(self, env):
        r1 = env.reset()
        r2 = env.reset()
        assert r1 == r2


# ── Step ────────────────────────────────────────────────────────

class TestStep:
    def test_step_return_signature(self, env):
        env.reset()
        result = env.step(0, 0)
        assert len(result) == 6, "step() should return (ro, bo, rr, br, done, info)"

    def test_step_obs_are_tuples(self, env):
        env.reset()
        ro, bo, rr, br, done, info = env.step(0, 0)
        assert isinstance(ro, tuple)
        assert isinstance(bo, tuple)

    def test_step_rewards_are_floats(self, env):
        env.reset()
        _, _, rr, br, _, _ = env.step(0, 0)
        assert isinstance(rr, float)
        assert isinstance(br, float)

    def test_step_done_false_before_max(self, env):
        env.reset()
        for s in range(env.max_steps - 1):
            _, _, _, _, done, _ = env.step(0, 0)
            assert not done, f"done=True at step {s+1} before max_steps={env.max_steps}"

    def test_step_done_true_at_max(self, env):
        env.reset()
        done = False
        for _ in range(env.max_steps):
            _, _, _, _, done, _ = env.step(0, 0)
        assert done, "done should be True after max_steps"

    def test_step_count_increments(self, env):
        env.reset()
        for i in range(1, 5):
            env.step(0, 0)
            assert env.step_count == i

    def test_invalid_red_action_raises(self, env):
        env.reset()
        with pytest.raises(AssertionError):
            env.step(NUM_ATTACKS, 0)  # out of bounds
        with pytest.raises(AssertionError):
            env.step(-1, 0)

    def test_invalid_blue_action_raises(self, env):
        env.reset()
        with pytest.raises(AssertionError):
            env.step(0, NUM_DEFENSES)
        with pytest.raises(AssertionError):
            env.step(0, -1)

    def test_all_valid_action_pairs_work(self, env):
        """All (attack, defense) combinations must not raise."""
        for ra in range(NUM_ATTACKS):
            for ba in range(NUM_DEFENSES):
                env.reset()
                env.step(ra, ba)   # must not raise

    def test_history_grows_with_steps(self, env):
        env.reset()
        for i in range(4):
            env.step(i % NUM_ATTACKS, i % NUM_DEFENSES)
        assert len(env.history) == 4


# ── Info dict ────────────────────────────────────────────────────

class TestInfoDict:
    REQUIRED_KEYS = {
        "step", "red_action", "blue_action", "attack_name", "attack_short",
        "defense_name", "defense_short", "outcome", "outcome_label",
        "p_success", "red_reward", "blue_reward", "blue_streak",
        "red_fail_streak", "attack_owasp", "defense_desc",
    }

    def test_info_has_required_keys(self, env):
        env.reset()
        _, _, _, _, _, info = env.step(0, 0)
        missing = self.REQUIRED_KEYS - set(info.keys())
        assert not missing, f"Info dict missing keys: {missing}"

    def test_info_outcome_in_valid_range(self, env):
        env.reset()
        for ra in range(NUM_ATTACKS):
            env.reset()
            _, _, _, _, _, info = env.step(ra, 0)
            assert info["outcome"] in (0, 1, 2), \
                f"outcome={info['outcome']} not in {{0,1,2}}"

    def test_info_outcome_label_matches_outcome(self, env):
        labels = {0: "BLOCKED", 1: "PARTIAL", 2: "JAILBREAK"}
        env.reset()
        for _ in range(10):
            _, _, _, _, done, info = env.step(
                random.randint(0, NUM_ATTACKS - 1),
                random.randint(0, NUM_DEFENSES - 1)
            )
            expected = labels[info["outcome"]]
            assert info["outcome_label"] == expected, \
                f"outcome={info['outcome']} but label='{info['outcome_label']}'"
            if done:
                break

    def test_info_p_success_in_range(self, env):
        env.reset()
        _, _, _, _, _, info = env.step(0, 0)
        assert 0.0 <= info["p_success"] <= 1.0

    def test_info_rewards_match_return_values(self, env):
        env.reset()
        _, _, rr, br, _, info = env.step(0, 0)
        assert info["red_reward"] == pytest.approx(rr, abs=1e-6)
        assert info["blue_reward"] == pytest.approx(br, abs=1e-6)

    def test_info_step_matches_env_step_count(self, env):
        env.reset()
        for expected_step in range(1, 5):
            _, _, _, _, _, info = env.step(0, 0)
            assert info["step"] == expected_step == env.step_count


# ── Reward semantics ─────────────────────────────────────────────

class TestRewards:
    def _run_until_outcome(self, env_inst, target_outcome, max_tries=500):
        """Force a specific outcome by retrying with fresh random seed."""
        import numpy as np
        for seed in range(max_tries):
            random.seed(seed)
            env_inst.reset()
            _, _, rr, br, _, info = env_inst.step(0, 0)
            if info["outcome"] == target_outcome:
                return rr, br, info
        pytest.skip(f"Could not produce outcome={target_outcome} in {max_tries} tries")

    def test_jailbreak_red_reward_positive(self, env20):
        rr, br, info = self._run_until_outcome(env20, LLMSecurityEnv.OUTCOME_JAILBREAK)
        assert rr > 0, f"Red should be rewarded for jailbreak, got rr={rr}"

    def test_jailbreak_blue_reward_negative(self, env20):
        rr, br, info = self._run_until_outcome(env20, LLMSecurityEnv.OUTCOME_JAILBREAK)
        assert br < 0, f"Blue should be penalized for jailbreak, got br={br}"

    def test_blocked_blue_reward_positive(self, env20):
        rr, br, info = self._run_until_outcome(env20, LLMSecurityEnv.OUTCOME_BLOCKED)
        assert br > 0, f"Blue should be rewarded for blocking, got br={br}"

    def test_blocked_red_reward_negative(self, env20):
        rr, br, info = self._run_until_outcome(env20, LLMSecurityEnv.OUTCOME_BLOCKED)
        assert rr < 0, f"Red should be penalized for being blocked, got rr={rr}"

    def test_rewards_are_zero_sum_ish(self, env20):
        """Red and Blue rewards should generally mirror each other (not exactly, due to bonuses)."""
        env20.reset()
        total_rr, total_br = 0.0, 0.0
        for _ in range(env20.max_steps):
            _, _, rr, br, done, _ = env20.step(
                random.randint(0, NUM_ATTACKS - 1),
                random.randint(0, NUM_DEFENSES - 1)
            )
            total_rr += rr
            total_br += br
            if done:
                break
        # Rewards are not always zero-sum due to bonuses, but should correlate negatively
        assert total_rr * total_br <= 50.0, \
            "Both agents shouldn't accumulate huge positive rewards simultaneously"

    def test_blue_streak_bonus_applied(self, env20):
        """After 3 consecutive blocks, Blue should get +0.4 bonus."""
        # Force blocks by using sanitizer (defense 0) vs FlipAttack (attack 0, SNTZ = 0.20 succeed)
        for seed in range(500):
            random.seed(seed)
            env20.reset()
            blocks = 0
            streak_bonus_seen = False
            for _ in range(env20.max_steps):
                _, _, rr, br, done, info = env20.step(0, 0)
                if info["outcome"] == LLMSecurityEnv.OUTCOME_BLOCKED:
                    blocks += 1
                    if blocks >= 3 and br > LLMSecurityEnv.BLUE_REWARD[0] + 0.35:
                        streak_bonus_seen = True
                        break
                else:
                    blocks = 0
                if done:
                    break
            if streak_bonus_seen:
                break
        assert streak_bonus_seen, "Blue streak bonus (+0.4) was never observed in 500 seeds"

    def test_novelty_bonus_on_first_use(self, env20):
        """Red should get +0.1 bonus for using a technique for the first time."""
        random.seed(42)
        env20.reset()
        _, _, rr, _, _, info = env20.step(1, 0)   # use attack 1 (shouldn't be in history yet)
        if info["outcome"] == LLMSecurityEnv.OUTCOME_BLOCKED:
            assert rr >= LLMSecurityEnv.RED_REWARD[0] + 0.1 - 0.001, \
                f"Expected novelty bonus +0.1 on first use, got rr={rr}"


# ── Metrics ──────────────────────────────────────────────────────

class TestMetrics:
    def test_empty_metrics(self, env):
        env.reset()
        assert env.get_metrics() == {}

    def test_rates_sum_to_one(self, env20):
        env20.reset()
        for _ in range(env20.max_steps):
            _, _, _, _, done, _ = env20.step(
                random.randint(0, NUM_ATTACKS - 1),
                random.randint(0, NUM_DEFENSES - 1)
            )
            if done:
                break
        m = env20.get_metrics()
        total = m["jailbreak_rate"] + m["partial_rate"] + m["block_rate"]
        assert abs(total - 1.0) < 0.01, f"Rates sum to {total}, expected ~1.0"

    def test_metrics_keys_present(self, env20):
        env20.reset()
        env20.step(0, 0)
        m = env20.get_metrics()
        for key in ("jailbreak_rate", "partial_rate", "block_rate",
                    "total_steps", "red_score", "blue_score"):
            assert key in m, f"Missing metric key: {key}"

    def test_total_steps_is_len_of_history(self, env20):
        env20.reset()
        n = 7
        for i in range(n):
            _, _, _, _, done, _ = env20.step(i % NUM_ATTACKS, i % NUM_DEFENSES)
            if done:
                n = i + 1
                break
        m = env20.get_metrics()
        assert m["total_steps"] == len(env20.history) == n

    def test_scores_match_accumulated_rewards(self, env20):
        env20.reset()
        cum_r, cum_b = 0.0, 0.0
        for _ in range(env20.max_steps):
            _, _, rr, br, done, _ = env20.step(
                random.randint(0, NUM_ATTACKS - 1),
                random.randint(0, NUM_DEFENSES - 1)
            )
            cum_r += rr
            cum_b += br
            if done:
                break
        m = env20.get_metrics()
        assert abs(m["red_score"] - round(cum_r, 3)) < 0.01
        assert abs(m["blue_score"] - round(cum_b, 3)) < 0.01


# ── Observation space ────────────────────────────────────────────

class TestObservations:
    def test_red_obs_caps_fail_streak_at_five(self, env20):
        """Red's fail streak observation is capped at 5."""
        env20.reset()
        # Force many blocks to accumulate streak
        for seed in range(200):
            random.seed(seed)
            env20.reset()
            for _ in range(env20.max_steps):
                ro, _, _, _, done, _ = env20.step(0, 0)
                assert ro[2] <= 5, f"fail_streak obs={ro[2]} exceeds cap of 5"
                if done:
                    break

    def test_blue_obs_caps_streak_at_five(self, env20):
        """Blue's block streak observation is capped at 5."""
        env20.reset()
        for seed in range(200):
            random.seed(seed)
            env20.reset()
            for _ in range(env20.max_steps):
                _, bo, _, _, done, _ = env20.step(0, 0)
                assert bo[2] <= 5, f"blue_streak obs={bo[2]} exceeds cap of 5"
                if done:
                    break

    def test_obs_step_increases_monotonically(self, env):
        env.reset()
        prev_step = -1
        for _ in range(env.max_steps):
            ro, bo, _, _, done, _ = env.step(0, 0)
            assert ro[3] >= prev_step
            assert bo[3] >= prev_step
            prev_step = ro[3]
            if done:
                break

    def test_obs_updates_after_step(self, env):
        """After step(ra, ba), blue obs should contain ra and red obs should contain ba."""
        env.reset()
        ra, ba = 3, 5
        ro, bo, _, _, _, _ = env.step(ra, ba)
        # blue obsrves red's action; red observes blue's action
        assert bo[0] == ra, f"Blue obs[0]={bo[0]} should reflect red_action={ra}"
        assert ro[0] == ba, f"Red obs[0]={ro[0]} should reflect blue_action={ba}"
