"""
tests/test_techniques.py
Unit tests for techniques.py — attack/defense taxonomy and interaction matrix.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from techniques import (
    RED_TECHNIQUES, BLUE_STRATEGIES, INTERACTION_MATRIX,
    NUM_ATTACKS, NUM_DEFENSES,
    get_attack, get_defense, get_success_probability,
    AttackTechnique, DefenseStrategy,
)


class TestTechniquesConstants:
    def test_num_attacks_matches_list(self):
        assert NUM_ATTACKS == len(RED_TECHNIQUES), \
            f"NUM_ATTACKS={NUM_ATTACKS} != len(RED_TECHNIQUES)={len(RED_TECHNIQUES)}"

    def test_num_defenses_matches_list(self):
        assert NUM_DEFENSES == len(BLUE_STRATEGIES), \
            f"NUM_DEFENSES={NUM_DEFENSES} != len(BLUE_STRATEGIES)={len(BLUE_STRATEGIES)}"

    def test_num_attacks_is_nine(self):
        assert NUM_ATTACKS == 9, "Expected 9 attack techniques"

    def test_num_defenses_is_seven(self):
        assert NUM_DEFENSES == 7, "Expected 7 defense strategies"


class TestAttackTechniques:
    def test_all_ids_unique(self):
        ids = [t.id for t in RED_TECHNIQUES]
        assert len(ids) == len(set(ids)), "Duplicate attack IDs found"

    def test_ids_are_sequential_from_zero(self):
        ids = sorted(t.id for t in RED_TECHNIQUES)
        assert ids == list(range(NUM_ATTACKS)), "Attack IDs must be 0..N-1"

    def test_all_shorts_unique(self):
        shorts = [t.short for t in RED_TECHNIQUES]
        assert len(shorts) == len(set(shorts)), "Duplicate attack short-codes"

    def test_base_success_in_range(self):
        for t in RED_TECHNIQUES:
            assert 0.0 < t.base_success <= 1.0, \
                f"{t.name}: base_success={t.base_success} out of (0, 1]"

    def test_all_have_nonempty_description(self):
        for t in RED_TECHNIQUES:
            assert t.description.strip(), f"{t.name} has empty description"

    def test_all_have_owasp_tag(self):
        for t in RED_TECHNIQUES:
            assert t.owasp.startswith("LLM"), \
                f"{t.name}: OWASP tag '{t.owasp}' does not start with 'LLM'"

    def test_get_attack_returns_correct_item(self):
        for i, expected in enumerate(RED_TECHNIQUES):
            got = get_attack(i)
            assert got.id == expected.id
            assert got.name == expected.name

    def test_get_attack_out_of_bounds_raises(self):
        with pytest.raises(IndexError):
            get_attack(NUM_ATTACKS)
        with pytest.raises(IndexError):
            get_attack(-1)


class TestDefenseStrategies:
    def test_all_ids_unique(self):
        ids = [d.id for d in BLUE_STRATEGIES]
        assert len(ids) == len(set(ids)), "Duplicate defense IDs found"

    def test_ids_are_sequential_from_zero(self):
        ids = sorted(d.id for d in BLUE_STRATEGIES)
        assert ids == list(range(NUM_DEFENSES)), "Defense IDs must be 0..N-1"

    def test_all_shorts_unique(self):
        shorts = [d.short for d in BLUE_STRATEGIES]
        assert len(shorts) == len(set(shorts)), "Duplicate defense short-codes"

    def test_base_block_in_range(self):
        for d in BLUE_STRATEGIES:
            assert 0.0 < d.base_block <= 1.0, \
                f"{d.name}: base_block={d.base_block} out of (0, 1]"

    def test_all_have_nonempty_description(self):
        for d in BLUE_STRATEGIES:
            assert d.description.strip(), f"{d.name} has empty description"

    def test_coverage_references_valid_attack_ids(self):
        for d in BLUE_STRATEGIES:
            for aid in d.coverage:
                assert 0 <= aid < NUM_ATTACKS, \
                    f"{d.name}: coverage references invalid attack id {aid}"

    def test_get_defense_returns_correct_item(self):
        for i, expected in enumerate(BLUE_STRATEGIES):
            got = get_defense(i)
            assert got.id == expected.id
            assert got.name == expected.name

    def test_get_defense_out_of_bounds_raises(self):
        with pytest.raises(IndexError):
            get_defense(NUM_DEFENSES)
        with pytest.raises(IndexError):
            get_defense(-1)


class TestInteractionMatrix:
    def test_matrix_dimensions(self):
        assert len(INTERACTION_MATRIX) == NUM_ATTACKS, \
            f"Matrix has {len(INTERACTION_MATRIX)} rows, expected {NUM_ATTACKS}"
        for i, row in enumerate(INTERACTION_MATRIX):
            assert len(row) == NUM_DEFENSES, \
                f"Row {i} has {len(row)} columns, expected {NUM_DEFENSES}"

    def test_all_probabilities_in_range(self):
        for i, row in enumerate(INTERACTION_MATRIX):
            for j, p in enumerate(row):
                assert 0.0 <= p <= 1.0, \
                    f"Matrix[{i}][{j}]={p} out of [0, 1]"

    def test_get_success_probability_matches_matrix(self):
        for i in range(NUM_ATTACKS):
            for j in range(NUM_DEFENSES):
                assert get_success_probability(i, j) == INTERACTION_MATRIX[i][j], \
                    f"Mismatch at [{i}][{j}]"

    def test_no_all_zero_rows(self):
        for i, row in enumerate(INTERACTION_MATRIX):
            assert any(p > 0 for p in row), \
                f"Attack {i} has all-zero success probabilities (unreachable)"

    def test_gcg_has_highest_average_success(self):
        """GCG (index 8) should be the hardest attack overall."""
        avgs = [sum(row) / len(row) for row in INTERACTION_MATRIX]
        gcg_avg = avgs[8]
        assert gcg_avg == max(avgs), \
            f"GCG avg={gcg_avg:.3f} is not the max among {[f'{a:.3f}' for a in avgs]}"

    def test_harm_classifier_lowest_gcg_success(self):
        """HARM Classifier (index 6) should be lowest P against GCG (index 8)."""
        gcg_row = INTERACTION_MATRIX[8]
        min_idx = gcg_row.index(min(gcg_row))
        assert min_idx == 6, \
            f"Lowest GCG success defense is index {min_idx}, expected 6 (HARM Classifier)"
