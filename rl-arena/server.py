"""
server.py — FastAPI backend for the RL Red Team vs Blue Team arena.
Streams real-time training events over WebSocket, serves the frontend.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from agents import RedTeamAgent, BlueTeamAgent
from env import LLMSecurityEnv
from techniques import RED_TECHNIQUES, BLUE_STRATEGIES, INTERACTION_MATRIX, NUM_ATTACKS, NUM_DEFENSES

# ── Global state ─────────────────────────────────────────────
red_agent   = RedTeamAgent()
blue_agent  = BlueTeamAgent()
env         = LLMSecurityEnv(max_steps=20)
clients: list[WebSocket] = []

training_state = {
    "running":          False,
    "episode":          0,
    "total_episodes":   500,
    "red_wins":         0,
    "blue_wins":        0,
    "draws":            0,
    "red_total_reward": 0.0,
    "blue_total_reward":0.0,
}

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="RL Red vs Blue Arena")


# ── WebSocket broadcast ──────────────────────────────────────
async def broadcast(message: dict):
    disconnected = []
    for ws in clients:
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        clients.remove(ws)


# ── RL Training loop ─────────────────────────────────────────
async def run_training(n_episodes: int = 500, delay: float = 0.04):
    global red_agent, blue_agent, env, training_state

    red_agent   = RedTeamAgent()
    blue_agent  = BlueTeamAgent()
    training_state.update({
        "running": True, "episode": 0,
        "red_wins": 0, "blue_wins": 0, "draws": 0,
        "red_total_reward": 0.0, "blue_total_reward": 0.0,
    })

    await broadcast({"type": "training_start", "n_episodes": n_episodes})

    for ep in range(1, n_episodes + 1):
        if not training_state["running"]:
            break

        red_obs, blue_obs = env.reset()
        ep_red_reward = 0.0
        ep_blue_reward = 0.0
        ep_steps = []

        done = False
        while not done:
            ra = red_agent.select_action(red_obs)
            ba = blue_agent.select_action(blue_obs)

            new_red_obs, new_blue_obs, rr, br, done, info = env.step(ra, ba)

            red_agent.update(red_obs,  ra, rr, new_red_obs,  done)
            blue_agent.update(blue_obs, ba, br, new_blue_obs, done)

            red_obs   = new_red_obs
            blue_obs  = new_blue_obs
            ep_red_reward  += rr
            ep_blue_reward += br
            ep_steps.append(info)

            # Stream each step live (throttle for performance)
            if len(ep_steps) % 3 == 0 or done:
                await broadcast({
                    "type":   "step",
                    "episode": ep,
                    "step":   info["step"],
                    "info":   info,
                })
            await asyncio.sleep(0)   # yield to event loop

        # Episode end
        red_agent.decay_epsilon()
        blue_agent.decay_epsilon()
        red_agent.record_episode_reward(ep_red_reward)
        blue_agent.record_episode_reward(ep_blue_reward)

        metrics = env.get_metrics()
        winner  = "RED" if ep_red_reward > ep_blue_reward else "BLUE" if ep_blue_reward > ep_red_reward else "DRAW"
        if winner == "RED":  training_state["red_wins"]  += 1
        elif winner == "BLUE": training_state["blue_wins"] += 1
        else:                  training_state["draws"]     += 1

        training_state["episode"] = ep
        training_state["red_total_reward"]  += ep_red_reward
        training_state["blue_total_reward"] += ep_blue_reward

        await broadcast({
            "type":            "episode_end",
            "episode":         ep,
            "red_reward":      round(ep_red_reward, 3),
            "blue_reward":     round(ep_blue_reward, 3),
            "winner":          winner,
            "metrics":         metrics,
            "red_epsilon":     round(red_agent.epsilon, 4),
            "blue_epsilon":    round(blue_agent.epsilon, 4),
            "red_action_dist": red_agent.get_action_distribution(),
            "blue_action_dist":blue_agent.get_action_distribution(),
            "red_rewards":     red_agent.episode_rewards[-100:],
            "blue_rewards":    blue_agent.episode_rewards[-100:],
            "red_cumulative":  red_agent.cumulative_rewards[-1] if red_agent.cumulative_rewards else 0,
            "blue_cumulative": blue_agent.cumulative_rewards[-1] if blue_agent.cumulative_rewards else 0,
            "training_state":  training_state.copy(),
        })

        if ep % 50 == 0:
            await broadcast({
                "type":          "checkpoint",
                "episode":       ep,
                "red_q_matrix":  red_agent.get_q_matrix(),
                "blue_q_matrix": blue_agent.get_q_matrix(),
                "red_stats":     red_agent.stats(),
                "blue_stats":    blue_agent.stats(),
            })

        await asyncio.sleep(delay)

    training_state["running"] = False
    await broadcast({
        "type":          "training_end",
        "training_state": training_state.copy(),
        "red_rewards":   red_agent.episode_rewards,
        "blue_rewards":  blue_agent.episode_rewards,
        "red_q_matrix":  red_agent.get_q_matrix(),
        "blue_q_matrix": blue_agent.get_q_matrix(),
        "red_stats":     red_agent.stats(),
        "blue_stats":    blue_agent.stats(),
    })


# ── Routes ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    # Send initial meta payload
    await ws.send_text(json.dumps({
        "type": "meta",
        "attacks":    [{"id": t.id, "name": t.name, "short": t.short, "owasp": t.owasp, "desc": t.description} for t in RED_TECHNIQUES],
        "defenses":   [{"id": d.id, "name": d.name, "short": d.short, "desc": d.description} for d in BLUE_STRATEGIES],
        "matrix":     INTERACTION_MATRIX,
        "n_attacks":  NUM_ATTACKS,
        "n_defenses": NUM_DEFENSES,
    }))
    try:
        while True:
            await ws.receive_text()   # keep-alive
    except WebSocketDisconnect:
        clients.remove(ws)

@app.post("/start")
async def start_training(episodes: int = 500, delay: float = 0.04):
    if training_state["running"]:
        return JSONResponse({"status": "already_running"})
    asyncio.create_task(run_training(n_episodes=episodes, delay=delay))
    return JSONResponse({"status": "started", "episodes": episodes})

@app.post("/stop")
async def stop_training():
    training_state["running"] = False
    return JSONResponse({"status": "stopped"})

@app.get("/stats")
async def get_stats():
    return JSONResponse({
        "training_state":  training_state.copy(),
        "red_stats":       red_agent.stats(),
        "blue_stats":      blue_agent.stats(),
        "red_rewards":     red_agent.episode_rewards,
        "blue_rewards":    blue_agent.episode_rewards,
        "red_q_matrix":    red_agent.get_q_matrix(),
        "blue_q_matrix":   blue_agent.get_q_matrix(),
    })


# ── Entry ─────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=False)
