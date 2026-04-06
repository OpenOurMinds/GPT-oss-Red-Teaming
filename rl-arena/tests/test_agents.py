"""
tests/test_agents.py
Unit tests for agents.py — Q-Learning agents.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
import numpy as np
import pytest
from agents import QLearningAgent, RedTeamAgent, BlueTeamAgent
from techniques import NUM_ATTACKS, NUM_DEFENSES


@pytest.fixture
def red():
    random.seed(0)
    return RedTeamAgent()

@pytest.fixture
def blue():
    random.seed(0)
    return BlueTeamAgent()

@pytest.fixture
def agent():
    """Generic QLearningAgent with 4 actions for isolated testing."""
    random.seed(0)
    return QLearningAgent(n_actions=4, name="TestAgent")


# ── Construction ─────────────────────────────────────────────────

class TestConstruction:
    def test_red_agent_n_actions(self, red):
        assert red.n_actions == NUM_ATTACKS

    def test_blue_agent_n_actions(self, blue):
        assert blue.n_actions == NUM_DEFENSES

    def test_initial_epsilon_is_one(self, red):
        assert red.epsilon == pytest.approx(1.0)

    def test_initial_q_table_empty(self, red):
        assert len(red.Q) == 0, "Q-table should be empty before any state is visited"

    def test_action_counts_all_zero(self, red):
        assert np.all(red.action_counts == 0)

    def test_episode_rewards_empty(self, red):
        assert red.episode_rewards == []
        assert red.cumulative_rewards == []

    def test_red_alpha(self, red):
        assert red.alpha == pytest.approx(0.15)

    def test_blue_alpha(self, blue):
        assert blue.alpha == pytest.approx(0.12)

    def test_blue_gamma(self, blue):
        assert blue.gamma == pytest.approx(0.92)

    def test_name(self, red, blue):
        assert red.name == "Red Team"
        assert blue.name == "Blue Team"


# ── select_action ────────────────────────────────────────────────

class TestSelectAction:
    def test_action_in_valid_range(self, red):
        state = (0, 0, 0, 0)
        for _ in range(50):
            a = red.select_action(state)
            assert 0 <= a < red.n_actions, f"action {a} out of range [0, {red.n_actions})"

    def test_action_is_int(self, red):
        state = (0, 0, 0, 0)
        a = red.select_action(state)
        assert isinstance(a, int)

    def test_with_epsilon_one_fully_random(self, agent):
        """With ε=1.0 all actions should be possible."""
        agent.epsilon = 1.0
        state = (0, 0, 0, 0)
        seen = set()
        for _ in range(1000):
            seen.add(agent.select_action(state))
        assert len(seen) == agent.n_actions, \
            f"With ε=1 expected all {agent.n_actions} actions, got {seen}"

    def test_with_epsilon_zero_greedy(self, agent):
        """With ε=0, always picks greedy action."""
        state = (1, 2, 0, 5)
        # Manually set Q so action 2 is always best
        agent.Q[state] = np.array([0.1, 0.2, 0.9, 0.3], dtype=np.float32)
        agent.epsilon = 0.0
        for _ in range(20):
            assert agent.select_action(state) == 2

    def test_creates_q_entry_on_visit(self, agent):
        """Q entry is created when greedy path accesses Q[state].
        With epsilon=0 the greedy branch always runs, so the defaultdict entry is made."""
        state = (7, 7, 7, 7)
        assert state not in agent.Q
        agent.epsilon = 0.0   # force greedy path, which reads self.Q[state]
        agent.select_action(state)
        assert state in agent.Q, \
            "Q[state] should be created by defaultdict on greedy path access"

    def test_different_states_independent(self, agent):
        s1, s2 = (0, 0, 0, 0), (1, 1, 1, 1)
        agent.Q[s1][0] = 10.0   # s1 best = action 0
        agent.Q[s2][3] = 10.0   # s2 best = action 3
        agent.epsilon = 0.0
        assert agent.select_action(s1) == 0
        assert agent.select_action(s2) == 3


# ── update (Bellman) ─────────────────────────────────────────────

class TestUpdate:
    def test_update_changes_q_value(self, agent):
        state = (0, 0, 0, 0)
        next_state = (1, 0, 0, 1)
        before = agent.Q[state][0]
        agent.update(state, 0, 1.0, next_state, False)
        after = agent.Q[state][0]
        assert before != after, "Q-value should change after update"

    def test_positive_reward_increases_q(self, agent):
        state = (0, 0, 0, 0)
        agent.Q[state] = np.zeros(4, dtype=np.float32)
        agent.update(state, 0, 1.0, (0, 0, 0, 1), True)
        assert agent.Q[state][0] > 0.0

    def test_negative_reward_decreases_q(self, agent):
        state = (0, 0, 0, 0)
        agent.Q[state] = np.zeros(4, dtype=np.float32)
        agent.update(state, 0, -1.0, (0, 0, 0, 1), True)
        assert agent.Q[state][0] < 0.0

    def test_done_ignores_next_state(self, agent):
        state = (0, 0, 0, 0)
        # Make next_state Q-values very large — should be ignored when done=True
        next_s = (9, 9, 9, 9)
        agent.Q[next_s] = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
        agent.Q[state] = np.zeros(4, dtype=np.float32)
        # done=True: target = reward only
        agent.update(state, 0, 1.0, next_s, done=True)
        expected = 0.0 + agent.alpha * (1.0 - 0.0)
        assert agent.Q[state][0] == pytest.approx(expected, abs=1e-5)

    def test_bellman_formula_exact(self, agent):
        """Verify Q(s,a) ← Q(s,a) + α*(r + γ*max_Q(s') - Q(s,a))"""
        state  = (2, 1, 0, 3)
        next_s = (3, 2, 0, 4)
        agent.Q[state]  = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        agent.Q[next_s] = np.array([0.2, 0.8, 0.1, 0.3], dtype=np.float32)
        reward = 0.7
        agent.update(state, 0, reward, next_s, done=False)
        expected = 0.5 + agent.alpha * (reward + agent.gamma * 0.8 - 0.5)
        assert agent.Q[state][0] == pytest.approx(expected, abs=1e-5)

    def test_action_count_increments(self, agent):
        state = (0, 0, 0, 0)
        for _ in range(5):
            agent.update(state, 2, 0.0, state, False)
        assert agent.action_counts[2] == 5
        assert agent.action_counts[0] == 0

    def test_update_does_not_affect_other_actions(self, agent):
        state = (0, 0, 0, 0)
        agent.Q[state] = np.zeros(4, dtype=np.float32)
        initial = agent.Q[state].copy()
        agent.update(state, 1, 1.0, state, True)
        for i in [0, 2, 3]:
            assert agent.Q[state][i] == initial[i], \
                f"Action {i} Q-value changed unexpectedly"


# ── decay_epsilon ──────────────────────────────────────────────

class TestEpsilonDecay:
    def test_epsilon_decreases_each_call(self, agent):
        prev = agent.epsilon
        for _ in range(20):
            agent.decay_epsilon()
            assert agent.epsilon <= prev
            prev = agent.epsilon

    def test_epsilon_does_not_go_below_min(self, agent):
        for _ in range(10000):
            agent.decay_epsilon()
        assert agent.epsilon >= agent.epsilon_min

    def test_epsilon_minimum_exact(self, agent):
        # Manually force many decays
        agent.epsilon = 1.0
        for _ in range(5000):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(agent.epsilon_min, abs=1e-6)

    def test_epsilon_formula(self, agent):
        e0 = agent.epsilon
        agent.decay_epsilon()
        expected = max(agent.epsilon_min, e0 * agent.epsilon_decay)
        assert agent.epsilon == pytest.approx(expected, abs=1e-8)


# ── record_episode_reward ────────────────────────────────────────

class TestEpisodeRewards:
    def test_record_appends_episode_reward(self, agent):
        agent.record_episode_reward(5.0)
        assert agent.episode_rewards[-1] == pytest.approx(5.0, abs=1e-4)

    def test_record_updates_cumulative(self, agent):
        agent.record_episode_reward(3.0)
        agent.record_episode_reward(2.0)
        assert agent.cumulative_rewards[-1] == pytest.approx(5.0, abs=1e-3)

    def test_cumulative_monotonically_increases_with_positive_rewards(self, agent):
        for v in [1.0, 2.5, 0.3, 4.0]:
            agent.record_episode_reward(v)
        for i in range(1, len(agent.cumulative_rewards)):
            assert agent.cumulative_rewards[i] >= agent.cumulative_rewards[i-1]

    def test_cumulative_can_decrease_with_negative_rewards(self, agent):
        agent.record_episode_reward(5.0)
        agent.record_episode_reward(-10.0)
        assert agent.cumulative_rewards[-1] < agent.cumulative_rewards[-2]

    def test_episode_reward_rounded_to_4dp(self, agent):
        agent.record_episode_reward(1.23456789)
        assert agent.episode_rewards[-1] == round(1.23456789, 4)


# ── get_action_distribution ──────────────────────────────────────

class TestActionDistribution:
    def test_sums_to_one_after_updates(self, agent):
        state = (0, 0, 0, 0)
        for a in range(agent.n_actions):
            for _ in range(10):
                agent.update(state, a, 0.1, state, False)
        dist = agent.get_action_distribution()
        assert abs(sum(dist) - 1.0) < 1e-4, f"Distribution sums to {sum(dist)}"

    def test_uniform_before_any_updates(self, agent):
        dist = agent.get_action_distribution()
        expected = 1.0 / agent.n_actions
        for p in dist:
            assert abs(p - expected) < 1e-6

    def test_length_matches_n_actions(self, agent):
        dist = agent.get_action_distribution()
        assert len(dist) == agent.n_actions

    def test_all_values_nonnegative(self, agent):
        dist = agent.get_action_distribution()
        assert all(p >= 0 for p in dist)


# ── get_q_matrix ──────────────────────────────────────────────────

class TestQMatrix:
    def test_returns_list_of_lists(self, agent):
        state = (0, 0, 0, 0)
        agent.update(state, 0, 1.0, state, True)
        m = agent.get_q_matrix()
        assert isinstance(m, list)
        assert isinstance(m[0], list)

    def test_each_row_length_matches_n_actions(self, agent):
        for s in [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)]:
            agent.update(s, 0, 0.5, s, True)
        m = agent.get_q_matrix()
        for row in m:
            assert len(row) == agent.n_actions

    def test_empty_q_table_returns_empty(self, agent):
        m = agent.get_q_matrix()
        assert m == [], "Empty Q-table should return empty matrix"


# ── stats ─────────────────────────────────────────────────────────

class TestStats:
    def test_stats_keys_present(self, red):
        s = red.stats()
        for key in ("name", "epsilon", "states_visited", "total_updates", "action_dist"):
            assert key in s, f"Missing key '{key}' in stats()"

    def test_stats_states_visited_matches_q_len(self, red):
        for i in range(5):
            state = (i, 0, 0, 0)
            red.select_action(state)
        s = red.stats()
        assert s["states_visited"] == len(red.Q)

    def test_stats_total_updates_matches_action_counts(self, red):
        state = (0, 0, 0, 0)
        for _ in range(7):
            red.update(state, 0, 0.1, state, False)
        s = red.stats()
        assert s["total_updates"] == int(red.action_counts.sum())


# ── Full episode integration ──────────────────────────────────────

class TestFullEpisodeIntegration:
    def test_full_training_episode(self):
        """Run 10 full episodes of env + both agents without errors."""
        from env import LLMSecurityEnv
        random.seed(99)
        env = LLMSecurityEnv(max_steps=15)
        red = RedTeamAgent()
        blue = BlueTeamAgent()

        for ep in range(10):
            ro, bo = env.reset()
            ep_rr = ep_br = 0.0
            done = False
            while not done:
                ra = red.select_action(ro)
                ba = blue.select_action(bo)
                new_ro, new_bo, rr, br, done, info = env.step(ra, ba)
                red.update(ro, ra, rr, new_ro, done)
                blue.update(bo, ba, br, new_bo, done)
                ro, bo = new_ro, new_bo
                ep_rr += rr
                ep_br += br
            red.decay_epsilon()
            blue.decay_epsilon()
            red.record_episode_reward(ep_rr)
            blue.record_episode_reward(ep_br)

        assert len(red.episode_rewards) == 10
        assert len(blue.episode_rewards) == 10
        assert red.epsilon < 1.0
        assert blue.epsilon < 1.0

    def test_q_learning_improves_over_episodes(self):
        """After many episodes, check that agents have learned (Q-tables non-trivial)."""
        from env import LLMSecurityEnv
        random.seed(7)
        env = LLMSecurityEnv(max_steps=20)
        red = RedTeamAgent()
        blue = BlueTeamAgent()

        for _ in range(100):
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

        # After 100 episodes agents should have visited many states
        assert len(red.Q) > 5, f"Red only visited {len(red.Q)} states after 100 eps"
        assert len(blue.Q) > 5, f"Blue only visited {len(blue.Q)} states after 100 eps"

        # Epsilon should have decayed significantly
        assert red.epsilon < 0.7
        assert blue.epsilon < 0.7
