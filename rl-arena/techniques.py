"""
techniques.py — SOTA Red Team attack vectors & Blue Team defense strategies
for the RL adversarial environment.
"""
from dataclasses import dataclass, field
from typing import List, Dict

# ─────────────────────────────────────────────
#  RED TEAM ATTACK TECHNIQUES
# ─────────────────────────────────────────────

@dataclass
class AttackTechnique:
    id: int
    name: str
    short: str          # For UI labels
    base_success: float # Base probability of bypassing a naive model
    description: str
    owasp: str          # Mapped OWASP LLM risk

RED_TECHNIQUES: List[AttackTechnique] = [
    AttackTechnique(0, "FlipAttack",          "FLIP",  0.88, "Reverses character order to confuse tokenization",                              "LLM01"),
    AttackTechnique(1, "TokenBreak",          "TBRK",  0.80, "Exploits BPE tokenization via Unicode substitution",                           "LLM01"),
    AttackTechnique(2, "DAN Injection",       "DAN",   0.72, "Developer-mode persona adoption to lift restrictions",                          "LLM01"),
    AttackTechnique(3, "LARGO Latent",        "LRGO",  0.85, "Latent-space embedding optimization for adversarial prompts",                   "LLM01"),
    AttackTechnique(4, "MIST Semantic",       "MIST",  0.78, "Iterative semantic refinement for black-box bypass",                            "LLM01"),
    AttackTechnique(5, "Role Reversal",       "ROLE",  0.65, "Convinces model it is the system developer with lifted restrictions",           "LLM01"),
    AttackTechnique(6, "Indirect Injection",  "INDR",  0.75, "Hides malicious instructions inside seemingly benign context",                  "LLM07"),
    AttackTechnique(7, "Contextual Logic",    "CNTX",  0.62, "Exploits long-context attention windows to bury harmful instructions",          "LLM01"),
    AttackTechnique(8, "GCG Adversarial",     "GCG",   0.90, "Greedy Coordinate Gradient — gradient-based universal suffix attack",          "LLM01"),
]

# ─────────────────────────────────────────────
#  BLUE TEAM DEFENSE STRATEGIES
# ─────────────────────────────────────────────

@dataclass
class DefenseStrategy:
    id: int
    name: str
    short: str
    base_block: float   # Base probability of blocking an attack
    coverage: List[int] # Attack technique IDs this defense targets most
    description: str

BLUE_STRATEGIES: List[DefenseStrategy] = [
    DefenseStrategy(0, "Input Sanitizer",    "SNTZ", 0.75, [0, 1],    "Strips Unicode lookalikes, invisible chars, and character reversals"),
    DefenseStrategy(1, "Llama Guard",        "LLMG", 0.82, [2, 5, 3], "Model-based output safety evaluation (Llama-Guard-3-8B category check)"),
    DefenseStrategy(2, "Content Filter",     "CNTF", 0.70, [4, 6],    "Keyword + semantic pattern matching against harmful content taxonomy"),
    DefenseStrategy(3, "Rate Limiter",       "RATE", 0.60, [4, 7],    "Adaptive query-rate throttling to disrupt multi-turn iterative attacks"),
    DefenseStrategy(4, "Prompt Rewriter",    "PRWR", 0.78, [6, 7],    "Paraphrases user input to break embedded injection context"),
    DefenseStrategy(5, "Delimiter Shield",   "DELM", 0.85, [6, 7, 2], "XML-wraps user data to enforce instruction/data separation"),
    DefenseStrategy(6, "Harm Classifier",    "HARM", 0.88, [8, 3, 1], "Embedding-based semantic harm classification (real-time inference)"),
]

# ─────────────────────────────────────────────
#  INTERACTION MATRIX
#  P(attack i succeeds | defense j active)
# ─────────────────────────────────────────────

# Dimensions: [9 attacks × 7 defenses]
# Lower = defense is more effective against that attack
INTERACTION_MATRIX = [
    # SNTZ  LLMG  CNTF  RATE  PRWR  DELM  HARM
    [0.20,  0.55,  0.65,  0.70,  0.60,  0.65,  0.45],  # FlipAttack
    [0.18,  0.48,  0.60,  0.68,  0.55,  0.58,  0.30],  # TokenBreak
    [0.65,  0.22,  0.50,  0.55,  0.60,  0.28,  0.40],  # DAN Injection
    [0.70,  0.30,  0.58,  0.62,  0.65,  0.70,  0.25],  # LARGO Latent
    [0.65,  0.55,  0.35,  0.30,  0.55,  0.65,  0.45],  # MIST Semantic
    [0.62,  0.28,  0.55,  0.58,  0.62,  0.25,  0.50],  # Role Reversal
    [0.60,  0.55,  0.35,  0.40,  0.25,  0.20,  0.52],  # Indirect Injection
    [0.65,  0.58,  0.48,  0.28,  0.22,  0.18,  0.55],  # Contextual Logic
    [0.72,  0.68,  0.72,  0.75,  0.68,  0.72,  0.20],  # GCG Adversarial
]

def get_success_probability(attack_id: int, defense_id: int) -> float:
    """Returns P(attack succeeds | defense active), factoring in base rates."""
    return INTERACTION_MATRIX[attack_id][defense_id]

def get_attack(i: int) -> AttackTechnique:
    if not (0 <= i < len(RED_TECHNIQUES)):
        raise IndexError(f"Attack index {i} out of range [0, {len(RED_TECHNIQUES)})")
    return RED_TECHNIQUES[i]

def get_defense(j: int) -> DefenseStrategy:
    if not (0 <= j < len(BLUE_STRATEGIES)):
        raise IndexError(f"Defense index {j} out of range [0, {len(BLUE_STRATEGIES)})")
    return BLUE_STRATEGIES[j]

NUM_ATTACKS = len(RED_TECHNIQUES)
NUM_DEFENSES = len(BLUE_STRATEGIES)
