import math
import os
import random

import numpy as np
import wandb

class TrainingController:
    def __init__(self, total_episodes=200, ball_count=15, random_balls=False, opponent_type="random", resume_checkpoint=False):
        self.ready = False
        self.init_error = None
        self.closed = False
        self.env = None
        self.wandb_run = None
        self.random_balls = bool(random_balls)
        self.max_ball_count = max(1, min(15, int(ball_count)))
        self.opponent_type = opponent_type
        self.resume_checkpoint = bool(resume_checkpoint)
        try:
            from enviorment import PoolEnvironment
            from ppo import PPOConfig, PPO_Agent, obs_to_compact_state
            import torch
        except Exception as exc:
            self.init_error = f"Training init failed: {exc}"
            return

        self.PoolEnvironment = PoolEnvironment
        self.PPOConfig = PPOConfig
        self.PPO_Agent = PPO_Agent
        self.obs_to_compact_state = obs_to_compact_state
        self.torch = torch

        self.curriculum = self._build_curriculum(self.max_ball_count, self.random_balls)
        self.curriculum_stage = 0
        start_stage = self.curriculum[self.curriculum_stage]
        self.env = self.PoolEnvironment(
            render_mode="rgb_array",
            ball_count=start_stage["ball_count"],
            random_balls=start_stage["random_balls"]
        )

        self.obs = self.env.reset()
        self.state_dim = self.obs_to_compact_state(self.obs, self.env.width, self.env.height).shape[0]

        self.cfg = self.PPOConfig()
        self.cfg.lr = 1e-4
        self.cfg.epochs = 6
        self.cfg.batch_size = 128
        self.cfg.rollout_steps = 512
        self.cfg.entropy_coef = 0.005
        self.agent = self.PPO_Agent(self.state_dim, self.cfg)

        self.total_episodes = total_episodes
        self.ep = 0
        self.games = 0
        self.wins = 0
        self.history = {"episode_reward": [], "win_rate": [], "eval_win_rate": []}
        self.global_step = 0
        self.ep_update_stats = []

        self.ep_reward = 0.0
        self.done = False
        self.paused = False

        self.buffer = {"states": [], "actions_tanh": [], "logp": [], "values": [], "rewards": [], "dones": []}
        self.eval_interval = 20
        self.eval_episodes = 12
        self.best_eval_win_rate = 0.0
        self.last_eval_win_rate = 0.0
        self.recent_results = []

        self.checkpoint_path = os.path.join("checkpoints", "snooker_ppo_latest.pt")
        self.best_checkpoint_path = os.path.join("checkpoints", "snooker_ppo_best.pt")
        self._load_checkpoint_if_exists()
        self.wandb_run = self._init_wandb_run()

        self.ready = True

    def _init_wandb_run(self):
        try:
            project = os.getenv("WANDB_PROJECT", "snooker-ppo")
            # Let W&B pick the default account/team unless explicitly set.
            entity = os.getenv("WANDB_ENTITY") or None
            run = wandb.init(
                project=project,
                entity=entity,
                config={
                    "source": "mySnooker_training_mode",
                    "game_type": self._game_type_label(),
                    "number_of_balls": self.max_ball_count,
                    "lr": self.cfg.lr,
                    "gamma": self.cfg.gamma,
                    "lam": self.cfg.lam,
                    "clip_eps": self.cfg.clip_eps,
                    "epochs": self.cfg.epochs,
                    "batch_size": self.cfg.batch_size,
                    "rollout_steps": self.cfg.rollout_steps,
                    "entropy_coef": self.cfg.entropy_coef,
                    "value_coef": self.cfg.value_coef,
                    "max_grad_norm": self.cfg.max_grad_norm,
                    "train_episodes": self.total_episodes,
                    "eval_interval": self.eval_interval,
                    "eval_episodes": self.eval_episodes,
                    "ball_count": self.max_ball_count,
                    "random_balls": self.random_balls,
                    "opponent_type": self.opponent_type,
                },
                reinit=True,
            )
            print(f"W&B run: {run.entity}/{run.project}/{run.name}")
            return run
        except Exception as exc:
            print(f"W&B init failed: {exc}")
            return None

    def _game_type_label(self):
        if self.opponent_type == "self":
            return "ai_vs_ai"
        if self.opponent_type == "none":
            return "solo"
        return "ai_vs_random"

    def toggle_pause(self):
        self.paused = not self.paused

    def _build_curriculum(self, max_ball_count, final_random):
        stages = [
            {"ball_count": min(3, max_ball_count), "random_balls": False},
            {"ball_count": min(5, max_ball_count), "random_balls": False},
            {"ball_count": min(8, max_ball_count), "random_balls": False},
            {"ball_count": max_ball_count, "random_balls": bool(final_random)},
        ]
        compact = []
        for s in stages:
            if not compact or compact[-1] != s:
                compact.append(s)
        return compact

    def _set_stage_env(self, stage_idx):
        stage_idx = max(0, min(stage_idx, len(self.curriculum) - 1))
        if stage_idx == self.curriculum_stage:
            return
        self.curriculum_stage = stage_idx
        stage = self.curriculum[self.curriculum_stage]
        if self.env:
            self.env.close()
        self.env = self.PoolEnvironment(
            render_mode="rgb_array",
            ball_count=stage["ball_count"],
            random_balls=stage["random_balls"]
        )
        self.obs = self.env.reset()
        self.done = False
        self.ep_reward = 0.0
        self.buffer = {"states": [], "actions_tanh": [], "logp": [], "values": [], "rewards": [], "dones": []}

    def _maybe_advance_curriculum(self):
        if self.curriculum_stage >= len(self.curriculum) - 1:
            return
        if len(self.recent_results) < 20:
            return
        recent_win_rate = sum(self.recent_results[-20:]) / 20.0
        if recent_win_rate >= 0.60:
            self._set_stage_env(self.curriculum_stage + 1)
            self.recent_results = []

    def _checkpoint_payload(self):
        return {
            "model": self.agent.net.state_dict(),
            "optimizer": self.agent.opt.state_dict(),
            "reward_mean": self.agent.reward_mean,
            "reward_var": self.agent.reward_var,
            "reward_count": self.agent.reward_count,
            "ep": self.ep,
            "games": self.games,
            "wins": self.wins,
            "best_eval_win_rate": self.best_eval_win_rate,
            "curriculum_stage": self.curriculum_stage,
        }

    def _save_checkpoint(self, best=False):
        try:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            self.torch.save(self._checkpoint_payload(), self.checkpoint_path)
            if best:
                self.torch.save(self._checkpoint_payload(), self.best_checkpoint_path)
        except Exception:
            pass

    def _torch_state_is_finite(self, value):
        if self.torch.is_tensor(value):
            if value.is_floating_point() or value.is_complex():
                return bool(self.torch.isfinite(value).all().item())
            return True
        if isinstance(value, dict):
            return all(self._torch_state_is_finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(self._torch_state_is_finite(v) for v in value)
        return True

    def _load_checkpoint_if_exists(self):
        if not os.path.exists(self.checkpoint_path):
            return
        try:
            payload = self.torch.load(self.checkpoint_path, map_location=self.agent.device)
            model_state = payload.get("model")
            if not isinstance(model_state, dict) or not self._torch_state_is_finite(model_state):
                print(f"Skipping invalid checkpoint model: {self.checkpoint_path}")
                return

            self.agent.net.load_state_dict(model_state)

            optimizer_state = payload.get("optimizer")
            if isinstance(optimizer_state, dict) and self._torch_state_is_finite(optimizer_state):
                self.agent.opt.load_state_dict(optimizer_state)
            else:
                print("Skipping invalid checkpoint optimizer state.")

            for attr in ("reward_mean", "reward_var", "reward_count"):
                value = float(payload.get(attr, getattr(self.agent, attr)))
                if math.isfinite(value):
                    setattr(self.agent, attr, value)

            best_eval_win_rate = float(payload.get("best_eval_win_rate", self.best_eval_win_rate))
            if math.isfinite(best_eval_win_rate):
                self.best_eval_win_rate = best_eval_win_rate

            if self.resume_checkpoint:
                self.ep = int(payload.get("ep", self.ep))
                self.games = int(payload.get("games", self.games))
                self.wins = int(payload.get("wins", self.wins))
                loaded_stage = int(payload.get("curriculum_stage", self.curriculum_stage))
                self._set_stage_env(loaded_stage)
            else:
                print("Loaded checkpoint weights; starting a fresh training session at episode 0.")
        except Exception as exc:
            print(f"Checkpoint load skipped: {exc}")

    def _random_action(self):
        angle = random.uniform(0, 2 * math.pi)
        power = random.uniform(0, 25)
        return {"angle": angle, "power": power}

    def _finish_episode(self):
        self.games += 1
        win = 1 if self.env.winner == 0 else 0
        if win:
            self.wins += 1
        self.recent_results.append(win)
        if len(self.recent_results) > 40:
            self.recent_results = self.recent_results[-40:]

        self.history["episode_reward"].append(self.ep_reward)
        self.history["win_rate"].append(self.wins / self.games)
        self.ep += 1
        self._maybe_advance_curriculum()

        if self.ep_update_stats:
            mean_policy_loss = float(np.mean([s["policy_loss"] for s in self.ep_update_stats]))
            mean_value_loss = float(np.mean([s["value_loss"] for s in self.ep_update_stats]))
            mean_total_loss = float(np.mean([s["total_loss"] for s in self.ep_update_stats]))
            mean_entropy = float(np.mean([s["entropy"] for s in self.ep_update_stats]))
            mean_approx_kl = float(np.mean([s["approx_kl"] for s in self.ep_update_stats]))
            mean_clip_frac = float(np.mean([s["clip_frac"] for s in self.ep_update_stats]))
        else:
            mean_policy_loss = 0.0
            mean_value_loss = 0.0
            mean_total_loss = 0.0
            mean_entropy = 0.0
            mean_approx_kl = 0.0
            mean_clip_frac = 0.0

        p1_score = float(self.env.players[0].get("score", 0))
        p2_score = float(self.env.players[1].get("score", 0))
        p1_remaining = float(self.env.count_remaining_balls(0))
        p2_remaining = float(self.env.count_remaining_balls(1))
        winner = int(self.env.winner) if self.env.winner is not None else -1
        stage = self.curriculum[self.curriculum_stage]
        win_rate = self.history["win_rate"][-1]
        win_percentage = win_rate * 100.0

        if self.wandb_run is not None:
            wandb.log({
                "episode": self.ep,
                "global_step": self.global_step,
                "episode_reward": self.ep_reward,
                "reward": self.ep_reward,
                "loss": mean_total_loss,
                "wins": self.wins,
                "games": self.games,
                "num_wins": self.wins,
                "num_matches": self.games,
                "win_rate": win_rate,
                "win_percentage": win_percentage,
                "eval_win_rate": self.last_eval_win_rate,
                "best_eval_win_rate": self.best_eval_win_rate,
                "game_type": self._game_type_label(),
                "number_of_balls": stage["ball_count"],
                "player1_score": p1_score,
                "player2_score": p2_score,
                "player1_remaining_balls": p1_remaining,
                "player2_remaining_balls": p2_remaining,
                "winner": winner,
                "curriculum_stage": self.curriculum_stage + 1,
                "curriculum_ball_count": stage["ball_count"],
                "curriculum_random_balls": int(stage["random_balls"]),
                "train_policy_loss": mean_policy_loss,
                "train_value_loss": mean_value_loss,
                "train_total_loss": mean_total_loss,
                "train_entropy": mean_entropy,
                "train_approx_kl": mean_approx_kl,
                "train_clip_frac": mean_clip_frac,
            }, step=self.ep)

        self.ep_update_stats = []

        if self.ep % self.eval_interval == 0:
            self.last_eval_win_rate = self._evaluate_policy(self.eval_episodes)
            self.history["eval_win_rate"].append(self.last_eval_win_rate)
            is_best = self.last_eval_win_rate >= self.best_eval_win_rate
            if is_best:
                self.best_eval_win_rate = self.last_eval_win_rate
            self._save_checkpoint(best=is_best)
        elif self.ep % 5 == 0:
            self._save_checkpoint(best=False)

    def _reset_episode(self):
        self.obs = self.env.reset()
        self.ep_reward = 0.0
        self.done = False
        self.ep_update_stats = []

    def _evaluate_policy(self, episodes):
        eval_env = self.PoolEnvironment(
            render_mode=None,
            ball_count=self.curriculum[self.curriculum_stage]["ball_count"],
            random_balls=self.curriculum[self.curriculum_stage]["random_balls"]
        )
        wins = 0
        max_turns = 80
        for _ in range(episodes):
            obs = eval_env.reset()
            done = False
            turns = 0
            while not done and turns < max_turns:
                s_vec = self.obs_to_compact_state(obs, eval_env.width, eval_env.height)
                if eval_env.current_player == 0:
                    action, _, _, _ = self.agent.select_action(s_vec, deterministic=True)
                else:
                    if self.opponent_type == "self":
                        action, _, _, _ = self.agent.select_action(s_vec, deterministic=True)
                    elif self.opponent_type == "none":
                        eval_env.current_player = 0
                        eval_env.foul = False
                        eval_env.turn_ended = False
                        eval_env.first_ball_hit = None
                        turns += 1
                        continue
                    else:
                        action = self._random_action()
                obs, _, done, _ = eval_env.step(action)
                turns += 1
            if eval_env.winner == 0:
                wins += 1
        eval_env.close()
        return wins / max(1, episodes)

    def step(self):
        if not self.ready or self.paused:
            return
        if self.ep >= self.total_episodes:
            return

        if self.opponent_type == "none" and self.env.current_player == 1:
            self.env.current_player = 0
            self.env.foul = False
            self.env.turn_ended = False
            self.env.first_ball_hit = None

        if self.done:
            self._finish_episode()
            if self.ep >= self.total_episodes:
                return
            self._reset_episode()

        s_vec = self.obs_to_compact_state(self.obs, self.env.width, self.env.height)

        if self.env.current_player == 0:
            action, logp, val, a_tanh = self.agent.select_action(s_vec)
            next_obs, reward, done, _ = self.env.step(action)

            self.buffer["states"].append(s_vec)
            self.buffer["actions_tanh"].append(a_tanh)
            self.buffer["logp"].append(logp)
            self.buffer["values"].append(val)
            self.buffer["rewards"].append(float(reward))
            self.buffer["dones"].append(float(done))
        else:
            if self.opponent_type == "self":
                action, _, _, _ = self.agent.select_action(s_vec)
                next_obs, reward, done, _ = self.env.step(action)
            else:
                next_obs, reward, done, _ = self.env.step(self._random_action())
            if done and self.buffer["dones"]:
                self.buffer["dones"][-1] = 1.0

        self.obs = next_obs
        self.ep_reward += float(reward)
        self.done = done
        self.global_step += 1

        if len(self.buffer["states"]) >= self.cfg.rollout_steps or done:
            values = np.array(self.buffer["values"] + [0.0], dtype=np.float32)
            rewards = np.array(self.buffer["rewards"], dtype=np.float32)
            rewards = self.agent.normalize_rewards(rewards)
            dones = np.array(self.buffer["dones"], dtype=np.float32)
            adv, ret = self.agent._gae(rewards, values, dones)
            self.buffer["adv"] = adv.tolist()
            self.buffer["returns"] = ret.tolist()

            stats = self.agent.update(self.buffer)
            self.ep_update_stats.append(stats)

            self.buffer = {"states": [], "actions_tanh": [], "logp": [], "values": [], "rewards": [], "dones": []}

    def get_frame(self):
        if not self.ready:
            return None
        return self.env.render(mode="rgb_array")

    def get_status_lines(self):
        if self.init_error:
            return [self.init_error, "Press ESC to return to menu"]

        if not self.ready:
            return ["Training not ready", "Press ESC to return to menu"]

        status = "Paused" if self.paused else "Running"
        if self.ep >= self.total_episodes:
            status = "Finished"

        win_rate = 0.0 if self.games == 0 else self.wins / self.games
        stage = self.curriculum[self.curriculum_stage]
        lines = [
            f"Episode: {self.ep}/{self.total_episodes} | Status: {status}",
            f"Episode Reward: {self.ep_reward:.2f} | Win Rate: {win_rate:.2f}",
            f"Eval Win Rate: {self.last_eval_win_rate:.2f} | Best: {self.best_eval_win_rate:.2f}",
            f"Curriculum: stage {self.curriculum_stage + 1}/{len(self.curriculum)} | balls={stage['ball_count']} random={'On' if stage['random_balls'] else 'Off'}",
            f"Opponent: {self.opponent_type}",
            f"Checkpoint: {self.checkpoint_path}",
            "Controls: SPACE pause/resume | ESC back to menu"
        ]
        return lines

    def stop_with_error(self, exc):
        self.init_error = f"Training stopped: {exc}"
        self.ready = False
        self.paused = True

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.ready:
            self._save_checkpoint(best=False)
        if self.env is not None:
            self.env.close()
            self.env = None
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

    def get_observation(self):
        return {
            "cue_ball": (self.env.cue_ball.x, self.env.cue_ball.y),
            "balls": [(b.x, b.y, b.potted) for b in self.env.balls],
            "current_player": self.env.current_player
        }

    def is_done(self):
        return self.done
