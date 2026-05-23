# ppo_project.py
import math
import random
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb

from enviorment import PoolEnvironment


# ---------------------------
# 1) State compression (MDP)
# ---------------------------
def _encode_player_type(player_type) -> Tuple[float, float]:
    if player_type == "solid":
        return 1.0, 0.0
    if player_type == "striped":
        return 0.0, 1.0
    return 0.0, 0.0


def _should_hit_8_ball(balls: List[dict], player_type: str) -> float:
    if player_type not in ("solid", "striped"):
        return 0.0
    for b in balls:
        if b.get("potted", False):
            continue
        if b.get("number", 0) == 8:
            continue
        b_type = "striped" if b.get("is_striped", False) else "solid"
        if b_type == player_type:
            return 0.0
    return 1.0


def _count_remaining(balls: List[dict], player_type: str) -> float:
    if player_type not in ("solid", "striped"):
        return 1.0
    remaining = 0
    for b in balls:
        if b.get("potted", False) or b.get("number", 0) == 8:
            continue
        b_type = "striped" if b.get("is_striped", False) else "solid"
        if b_type == player_type:
            remaining += 1
    return float(remaining) / 7.0


def obs_to_compact_state(obs: dict, width: float, height: float, max_object_balls: int = 15) -> np.ndarray:
    """
    Compact MDP state:
    For each ball (ordered): (x_norm, y_norm, potted, is_striped, is_eight, is_cue)
    + current player
    + player types (current + opponent)
    + game flags (assignment_done, foul, turn_ended, winner)
    + should_hit_8_ball, remaining_balls
    """
    balls = obs["balls"]
    max_object_balls = max(1, min(15, int(max_object_balls)))

    cue_ball = None
    other_balls = []
    for b in balls:
        if b.get("number", 0) == 0:
            cue_ball = b
        else:
            other_balls.append(b)

    if cue_ball is None and balls:
        cue_ball = balls[0]
        other_balls = balls[1:]
    elif cue_ball is None:
        cue_ball = {
            "x": width * 0.25,
            "y": height * 0.5,
            "potted": False,
            "is_striped": False,
            "number": 0,
        }

    ordered_objects = sorted(other_balls, key=lambda b: b.get("number", 0))[:max_object_balls]
    while len(ordered_objects) < max_object_balls:
        ordered_objects.append({
            "x": 0.0,
            "y": 0.0,
            "potted": True,
            "is_striped": False,
            "number": -1,
        })

    ordered = [cue_ball] + ordered_objects

    vec = []
    for b in ordered:
        x = b["x"] / width
        y = b["y"] / height
        potted = 1.0 if b.get("potted", False) else 0.0
        is_striped = 1.0 if b.get("is_striped", False) else 0.0
        is_eight = 1.0 if b.get("number", 0) == 8 else 0.0
        is_cue = 1.0 if b.get("number", 0) == 0 else 0.0
        vec.extend([x, y, potted, is_striped, is_eight, is_cue])

    current_player = int(obs.get("current_player", 0))
    player_types = obs.get("player_types", [None, None])
    cur_type = player_types[current_player] if player_types else None
    opp_type = player_types[1 - current_player] if player_types and len(player_types) > 1 else None

    cur_solid, cur_striped = _encode_player_type(cur_type)
    opp_solid, opp_striped = _encode_player_type(opp_type)

    assignment_done = 1.0 if obs.get("ball_assignment_done", False) else 0.0
    foul = 1.0 if obs.get("foul", False) else 0.0
    turn_ended = 1.0 if obs.get("turn_ended", False) else 0.0
    winner = obs.get("winner", None)
    winner_p0 = 1.0 if winner == 0 else 0.0
    winner_p1 = 1.0 if winner == 1 else 0.0

    should_hit_8 = _should_hit_8_ball(balls, cur_type)
    remaining = _count_remaining(balls, cur_type)

    vec.extend([
        float(current_player),
        cur_solid, cur_striped,
        opp_solid, opp_striped,
        assignment_done, foul, turn_ended,
        winner_p0, winner_p1,
        should_hit_8, remaining
    ])

    return np.array(vec, dtype=np.float32)


# ---------------------------
# 2) PPO Network (class PPO)
# ---------------------------
class PPO(nn.Module):
    
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((2,), -0.7))

        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        mu = torch.tanh(self.mu(h))               # (-1,1)
        std = torch.exp(self.log_std).clamp(1e-3, 1.0)
        value = self.v(h).squeeze(-1)
        return mu, std, value
    
@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 1e-4
    epochs: int = 6
    batch_size: int = 128
    rollout_steps: int = 512
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    value_clip: float = 10.0
    reward_norm: bool = True
    reward_clip: float = 5.0
    reward_eps: float = 1e-8
    use_ppog_actor_loss: bool = True

    # action ranges based on your random agent
    angle_min: float = 0.0
    angle_max: float = 2 * math.pi
    power_min: float = 0.0
    power_max: float = 25.0


class PPO_Agent:
    def __init__(self, state_dim: int, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PPO(state_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = cfg.reward_eps

    def normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        if rewards.size == 0 or not self.cfg.reward_norm:
            return rewards

        batch_mean = float(np.mean(rewards))
        batch_var = float(np.var(rewards))
        batch_count = float(rewards.shape[0])

        delta = batch_mean - self.reward_mean
        total_count = self.reward_count + batch_count

        new_mean = self.reward_mean + delta * batch_count / total_count
        m_a = self.reward_var * self.reward_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.reward_count * batch_count / total_count
        new_var = m2 / total_count

        self.reward_mean = new_mean
        self.reward_var = max(new_var, self.cfg.reward_eps)
        self.reward_count = total_count

        normed = (rewards - self.reward_mean) / math.sqrt(self.reward_var + self.cfg.reward_eps)
        return np.clip(normed, -self.cfg.reward_clip, self.cfg.reward_clip).astype(np.float32)

    def _scale_action(self, a_tanh: np.ndarray) -> Dict[str, float]:
        
        u = (a_tanh + 1.0) / 2.0
        angle = self.cfg.angle_min + u[0] * (self.cfg.angle_max - self.cfg.angle_min)
        power = self.cfg.power_min + u[1] * (self.cfg.power_max - self.cfg.power_min)
        return {"angle": float(angle), "power": float(power)}

    def action_to_tanh(self, action: Dict[str, float]) -> np.ndarray:
        angle_range = self.cfg.angle_max - self.cfg.angle_min
        power_range = self.cfg.power_max - self.cfg.power_min
        angle = float(action.get("angle", self.cfg.angle_min))
        power = float(action.get("power", self.cfg.power_min))

        angle_u = (angle - self.cfg.angle_min) / angle_range
        power_u = (power - self.cfg.power_min) / power_range
        a_tanh = np.array([angle_u * 2.0 - 1.0, power_u * 2.0 - 1.0], dtype=np.float32)
        return np.clip(a_tanh, -0.999, 0.999)

    @torch.no_grad()
    def logp_value_for_action(self, state_vec: np.ndarray, action_tanh: np.ndarray) -> Tuple[float, float]:
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.tensor(action_tanh, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, std, v = self.net(s)
        dist = torch.distributions.Normal(mu, std)
        pre_tanh = torch.atanh(a.clamp(-0.999, 0.999))
        logp = dist.log_prob(pre_tanh).sum(dim=-1)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
        return float(logp.item()), float(v.item())

    @torch.no_grad()
    def select_action(self, state_vec: np.ndarray, deterministic: bool = False) -> Tuple[Dict[str, float], float, float, np.ndarray]:
       
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, std, v = self.net(s)
        dist = torch.distributions.Normal(mu, std)
        if deterministic:
            a = torch.tanh(mu)
            pre_tanh = torch.atanh(a.clamp(-0.999, 0.999))
        else:
            pre_tanh = dist.sample()
            a = torch.tanh(pre_tanh)
        logp = dist.log_prob(pre_tanh).sum(dim=-1)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)

        a_np = a.squeeze(0).cpu().numpy()
        action = self._scale_action(a_np)
        return action, float(logp.item()), float(v.item()), a_np

    @torch.no_grad()
    def value(self, state_vec: np.ndarray) -> float:
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, _, v = self.net(s)
        return float(v.item())

    def imitation_update(self, state_vec: np.ndarray, action_tanh: np.ndarray) -> float:
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        target = torch.tensor(action_tanh, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, _, _ = self.net(s)
        pred = torch.tanh(mu)
        loss = (pred - target).pow(2).mean()
        if not torch.isfinite(loss):
            return 0.0

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        return float(loss.item())

    def _gae(self, rewards, values, dones):
        cfg = self.cfg       
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            nextnonterminal = 1.0 - dones[t]
            nextvalue = values[t + 1] if t + 1 < len(values) else values[t]
            delta = rewards[t] + cfg.gamma * nextvalue * nextnonterminal - values[t]
            lastgaelam = delta + cfg.gamma * cfg.lam * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + values[:len(rewards)]
        return adv, ret

    def update(self, buffer):
        cfg = self.cfg
        if not buffer["states"]:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "total_loss": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
            }

        states = torch.tensor(np.array(buffer["states"]), dtype=torch.float32, device=self.device)
        actions_tanh = torch.tensor(np.array(buffer["actions_tanh"]), dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(np.array(buffer["logp"]), dtype=torch.float32, device=self.device)
        returns = torch.tensor(np.array(buffer["returns"]), dtype=torch.float32, device=self.device)
        adv = torch.tensor(np.array(buffer["adv"]), dtype=torch.float32, device=self.device)

        adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        adv_std = adv.std(unbiased=False)
        if torch.isfinite(adv_std) and adv_std > 1e-8:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)
        else:
            adv = adv - adv.mean()

        n = states.shape[0]
        idxs = np.arange(n)

        policy_losses = []
        value_losses = []
        entropies = []
        total_losses = []
        approx_kls = []
        clip_fracs = []

        for _ in range(cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, cfg.batch_size):
                batch = idxs[start:start + cfg.batch_size]

                s_b = states[batch]
                a_b = actions_tanh[batch]
                old_logp_b = old_logp[batch]
                ret_b = returns[batch]
                adv_b = adv[batch]

                mu, std, v = self.net(s_b)
                if not (torch.isfinite(mu).all() and torch.isfinite(std).all() and torch.isfinite(v).all()):
                    continue
                dist = torch.distributions.Normal(mu, std)

                # inverse tanh for log_prob calculation
                atanh_a = torch.atanh(a_b.clamp(-0.999, 0.999))
                logp = dist.log_prob(atanh_a).sum(dim=-1)
                logp -= torch.log(1 - a_b.pow(2) + 1e-6).sum(dim=-1)

                ratio = torch.exp(logp - old_logp_b)
                with torch.no_grad():
                    approx_kl = (old_logp_b - logp).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                if cfg.use_ppog_actor_loss:
                    # PPOG (Gilad version):
                    # if ratio is out of trust region in the "right" direction, stop actor update for that sample.
                    regular = -ratio * adv_b
                    out_hi = ratio > (1 + cfg.clip_eps)
                    out_lo = ratio < (1 - cfg.clip_eps)
                    right_dir = ((adv_b > 0) & out_hi) | ((adv_b < 0) & out_lo)
                    policy_terms = torch.where(right_dir, torch.zeros_like(regular), regular)
                    policy_loss = policy_terms.mean()
                else:
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()

                value_error = (ret_b - v).clamp(-cfg.value_clip, cfg.value_clip)
                value_loss = value_error.pow(2).mean()
                entropy = dist.entropy().sum(dim=-1).mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                if not torch.isfinite(loss):
                    continue

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    self.net.log_std.clamp_(-2.0, 0.0)

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                total_losses.append(float(loss.item()))
                approx_kls.append(float(approx_kl))
                clip_fracs.append(float(clip_frac))

            if approx_kls and float(np.mean(approx_kls[-max(1, math.ceil(n / cfg.batch_size)):])) > cfg.target_kl:
                break

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        }


class Trainer:
    def __init__(self, env: PoolEnvironment, agent: PPO_Agent, opponent_policy=None):
        self.env = env
        self.agent = agent
        self.opponent_policy = opponent_policy or self._random_action

        self.history = {
            "episode_reward": [],
            "win_rate": [],
        }

    def _random_action(self) -> Dict[str, float]:
        angle = random.uniform(0, 2 * math.pi)
        power = random.uniform(0, 25)
        return {"angle": angle, "power": power}

    def train(self, total_episodes: int = 200):
        cfg = self.agent.cfg
        wins = 0
        games = 0

        for ep in range(total_episodes):
            obs = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            update_stats = []

            buffer = {"states": [], "actions_tanh": [], "logp": [], "values": [], "rewards": [], "dones": []}

            while not done:
                
                s_vec = obs_to_compact_state(obs, self.env.width, self.env.height)

                if self.env.current_player == 0:
                    action, logp, val, a_tanh = self.agent.select_action(s_vec)


                    next_obs, reward, done, info = self.env.step(action)
                    agent_reward = float(reward)

                    buffer["states"].append(s_vec)
                    buffer["actions_tanh"].append(a_tanh)
                    buffer["logp"].append(logp)
                    buffer["values"].append(val)
                    buffer["rewards"].append(agent_reward)
                    buffer["dones"].append(float(done))

                else:
                    next_obs, reward, done, info = self.env.step(self.opponent_policy())
                    agent_reward = -float(reward)
                    if done and buffer["dones"]:
                        buffer["dones"][-1] = 1.0
                        buffer["rewards"][-1] += agent_reward

                obs = next_obs
                ep_reward += agent_reward
                ep_steps += 1

                if len(buffer["states"]) >= cfg.rollout_steps or done:
                    if done:
                        bootstrap_value = 0.0
                    else:
                        bootstrap_state = obs_to_compact_state(obs, self.env.width, self.env.height)
                        bootstrap_value = self.agent.value(bootstrap_state)
                    values = np.array(buffer["values"] + [bootstrap_value], dtype=np.float32)
                    rewards = np.array(buffer["rewards"], dtype=np.float32)
                    rewards = self.agent.normalize_rewards(rewards)
                    dones = np.array(buffer["dones"], dtype=np.float32)
                    adv, ret = self.agent._gae(rewards, values, dones)
                    buffer["adv"] = adv.tolist()
                    buffer["returns"] = ret.tolist()

                    stats = self.agent.update(buffer)
                    update_stats.append(stats)

                    buffer = {"states": [], "actions_tanh": [], "logp": [], "values": [], "rewards": [], "dones": []}

            games += 1
            if self.env.winner == 0:
                wins += 1

            self.history["episode_reward"].append(ep_reward)
            self.history["win_rate"].append(wins / games)
            p1_score = float(self.env.players[0].get("score", 0))
            p2_score = float(self.env.players[1].get("score", 0))
            p1_remaining = float(self.env.count_remaining_balls(0))
            p2_remaining = float(self.env.count_remaining_balls(1))
            winner = int(self.env.winner) if self.env.winner is not None else -1

            if update_stats:
                mean_policy_loss = float(np.mean([s["policy_loss"] for s in update_stats]))
                mean_value_loss = float(np.mean([s["value_loss"] for s in update_stats]))
                mean_entropy = float(np.mean([s["entropy"] for s in update_stats]))
                mean_total_loss = float(np.mean([s["total_loss"] for s in update_stats]))
                mean_approx_kl = float(np.mean([s["approx_kl"] for s in update_stats]))
                mean_clip_frac = float(np.mean([s["clip_frac"] for s in update_stats]))
            else:
                mean_policy_loss = 0.0
                mean_value_loss = 0.0
                mean_entropy = 0.0
                mean_total_loss = 0.0
                mean_approx_kl = 0.0
                mean_clip_frac = 0.0

            print(f"Episode {ep+1}/{total_episodes} | reward={ep_reward:.2f} | win_rate={self.history['win_rate'][-1]:.2f}")
            wandb.log({
                "episode": ep + 1,
                "episode_reward": ep_reward,
                "reward": ep_reward,
                "loss": mean_total_loss,
                "wins": wins,
                "games": games,
                "num_wins": wins,
                "num_matches": games,
                "win_rate": self.history["win_rate"][-1],
                "win_percentage": self.history["win_rate"][-1] * 100.0,
                "number_of_balls": self.env.ball_count,
                "game_type": "ai_vs_random",
                "episode_steps": ep_steps,
                "player1_score": p1_score,
                "player2_score": p2_score,
                "player1_remaining_balls": p1_remaining,
                "player2_remaining_balls": p2_remaining,
                "winner": winner,
                "train_policy_loss": mean_policy_loss,
                "train_value_loss": mean_value_loss,
                "train_entropy": mean_entropy,
                "train_total_loss": mean_total_loss,
                "train_approx_kl": mean_approx_kl,
                "train_clip_frac": mean_clip_frac,
            }, step=ep + 1)


# ---------------------------
# 5) Tester (class Tester)
# ---------------------------
class Tester:    
    def __init__(self, env: PoolEnvironment, agent: PPO_Agent, opponent_policy=None):
        self.env = env
        self.agent = agent
        self.opponent_policy = opponent_policy or (lambda: {"angle": random.uniform(0, 2 * math.pi), "power": random.uniform(0, 25)})

    def evaluate(self, episodes: int = 50) -> Dict[str, float]:
        wins = 0
        for _ in range(episodes):
            obs = self.env.reset()
            done = False
            while not done:
                s_vec = obs_to_compact_state(obs, self.env.width, self.env.height)
                if self.env.current_player == 0:
                    action, _, _, _ = self.agent.select_action(s_vec, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                else:
                    obs, reward, done, info = self.env.step(self.opponent_policy())

            if self.env.winner == 0:
                wins += 1

        return {"wins": wins, "episodes": episodes, "win_rate": wins / max(1, episodes)}


# ---------------------------
# 6) print_results (class)
# ---------------------------
class print_results:
    @staticmethod
    def plot(history: Dict[str, List[float]]):
        plt.figure()
        plt.plot(history["episode_reward"])
        plt.title("Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()

        plt.figure()
        plt.plot(history["win_rate"])
        plt.title("Win Rate vs Random")
        plt.xlabel("Episode")
        plt.ylabel("Win Rate")
        plt.ylim(0, 1)
        plt.show()


def init_wandb(cfg: PPOConfig, train_episodes: int, eval_episodes: int, ball_count: int, random_balls: bool):
    config = {
        "lr": cfg.lr,
        "gamma": cfg.gamma,
        "lam": cfg.lam,
        "clip_eps": cfg.clip_eps,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "rollout_steps": cfg.rollout_steps,
        "entropy_coef": cfg.entropy_coef,
        "value_coef": cfg.value_coef,
        "max_grad_norm": cfg.max_grad_norm,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "ball_count": ball_count,
        "number_of_balls": ball_count,
        "random_balls": random_balls,
        "game_type": "ai_vs_random",
    }

    mode = os.getenv("WANDB_MODE")
    api_key = os.getenv("WANDB_API_KEY", "")
    if mode is None and api_key and len(api_key) < 40:
        print(f"WANDB_API_KEY looks invalid (len={len(api_key)}). Disabling W&B logging.")
        mode = "disabled"

    try:
        if mode:
            return wandb.init(project="snooker-ppo", entity="noamtalfus1312-null", config=config, mode=mode)
        return wandb.init(project="snooker-ppo", entity="noamtalfus1312-null", config=config)
    except wandb.errors.AuthenticationError as exc:
        print(f"W&B auth failed ({exc}). Continuing with WANDB_MODE=disabled.")
        os.environ["WANDB_MODE"] = "disabled"
        return wandb.init(project="snooker-ppo", entity="noamtalfus1312-null", config=config, mode="disabled")


# ---------------------------
# Run
# ---------------------------
def main():
    train_episodes = 200
    eval_episodes = 50
    ball_count = 15
    random_balls = False

    env = PoolEnvironment(render_mode=None, ball_count=ball_count, random_balls=random_balls)
    # build one obs to infer state_dim
    obs = env.reset()
    state_dim = obs_to_compact_state(obs, env.width, env.height).shape[0]

    cfg = PPOConfig()
    run = init_wandb(cfg, train_episodes, eval_episodes, ball_count, random_balls)
    agent = PPO_Agent(state_dim, cfg)

    trainer = Trainer(env, agent)
    trainer.train(total_episodes=train_episodes)

    tester = Tester(env, agent)
    res = tester.evaluate(episodes=eval_episodes)
    print("TEST:", res)
    wandb.log({
        "test_win_rate": res["win_rate"],
        "test_wins": res["wins"],
        "test_episodes": res["episodes"],
    })

    print_results.plot(trainer.history)
    run.finish()


if __name__ == "__main__":
    main()
