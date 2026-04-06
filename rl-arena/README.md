# ⚔️ RL Arena — LLM Security Reinforcement Learning Environment

A real-time, adversarial reinforcement learning simulator where a **Red Team** (attacker) and **Blue Team** (defender) learn to outmaneuver each other in an LLM security context.

## 🚀 Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv_rl
source venv_rl/bin/activate

# 2. Install dependencies
pip install fastapi "uvicorn[standard]" numpy

# 3. Launch the arena server
python3 server.py
```

Then open **http://localhost:8765** in your browser and click **▶ START**.

---

## 🏗️ Architecture

```
rl-arena/
├── techniques.py   # 9 attack vectors × 7 defenses + 9×7 interaction matrix
├── env.py          # LLMSecurityEnv — Markov Decision Process (MDP)
├── agents.py       # RedTeamAgent + BlueTeamAgent (tabular Q-Learning)
├── server.py       # FastAPI + WebSocket streaming backend
└── index.html      # Monotone dark live dashboard (5-panel UI)
```

---

## 🔴 Red Team — Attack Techniques

| ID | Name | Short | Research Success Rate |
|---|---|---|---|
| 0 | FlipAttack | `FLIP` | 88% |
| 1 | TokenBreak (BPE) | `TBRK` | 80% |
| 2 | DAN Injection | `DAN` | 72% |
| 3 | LARGO Latent | `LRGO` | 85% |
| 4 | MIST Semantic | `MIST` | 78% |
| 5 | Role Reversal | `ROLE` | 65% |
| 6 | Indirect Injection | `INDR` | 75% |
| 7 | Contextual Logic | `CNTX` | 62% |
| 8 | **GCG Adversarial** | `GCG` | **90%** |

## 🔵 Blue Team — Defense Strategies

| ID | Name | Short | Block Rate |
|---|---|---|---|
| 0 | Input Sanitizer | `SNTZ` | 75% |
| 1 | Llama Guard | `LLMG` | 82% |
| 2 | Content Filter | `CNTF` | 70% |
| 3 | Rate Limiter | `RATE` | 60% |
| 4 | Prompt Rewriter | `PRWR` | 78% |
| 5 | Delimiter Shield | `DELM` | 85% |
| 6 | **Harm Classifier** | `HARM` | **88%** |

---

## 🧠 RL Algorithm

Both agents use **Tabular Q-Learning** with ε-greedy exploration:

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s', a') − Q(s,a)]
```

| Parameter | Red Agent | Blue Agent |
|---|---|---|
| Learning rate α | 0.15 | 0.12 |
| Discount γ | 0.90 | 0.92 |
| ε start | 1.00 | 1.00 |
| ε min | 0.05 | 0.05 |
| ε decay | 0.995/ep | 0.995/ep |

### Reward Structure

| Outcome | Red Reward | Blue Reward |
|---|---|---|
| JAILBREAK | +1.5 | −1.5 |
| PARTIAL | +0.3 | −0.3 |
| BLOCKED | −0.6 | +1.0 |
| Blue Streak (3+) | — | +0.4 bonus |
| Novel technique | +0.1 bonus | — |

---

## 📊 Dashboard Panels

1. **Live Battle Feed** — real-time step log: technique vs defense, outcome, Δreward
2. **Episode Reward Curves** — per-episode rewards for both teams (canvas chart)
3. **Scoreboard + Epsilon** — win counts, exploration rate meters, action distribution
4. **Q-Value Heatmap** — state-action Q-table heatmap (Red Q / Blue Q / Interaction Matrix)
5. **Cumulative Reward + Win Rate** — long-run learning trend and win percentage bars

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve dashboard UI |
| `WS` | `/ws` | Real-time training event stream |
| `POST` | `/start?episodes=N` | Begin RL training run |
| `POST` | `/stop` | Halt training |
| `GET` | `/stats` | Full Q-tables and metrics (JSON) |

---

## ⚠️ Ethical Notice

This environment is designed **exclusively for defensive AI safety research**. All techniques are implemented for educational and vulnerability-disclosure purposes only.
