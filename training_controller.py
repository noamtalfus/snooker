import math
import os
import random

import numpy as np
import wandb

from shot_guidance import guided_shot_from_observation

class TrainingController:
    def __init__(self, total_episodes=2000, ball_count=15, random_balls=False, opponent_type="random", resume_checkpoint=False, layout="rack", workout_plan=False):
        self.ready = False
        self.init_error = None
        self.closed = False
        self.env = None
        self.wandb_run = None
        self.total_episodes = total_episodes
        self.target_random_balls = bool(random_balls)
        self.target_ball_count = max(2, min(15, int(ball_count)))
        self.target_layout = self._normalize_layout(layout)
        self.target_opponent_type = opponent_type
        self.workout_plan = bool(workout_plan)
        self.random_balls = self.target_random_balls
        self.max_ball_count = self.target_ball_count
        self.layout = self.target_layout
        self.state_object_balls = 15
        self.opponent_type = self.target_opponent_type
        self.curriculum = []
        self.curriculum_stage = 0
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

        self.curriculum = self._build_workout_plan() if self.workout_plan else [self._target_stage("Selected Settings", self.total_episodes)]
        self._apply_curriculum_stage(0, create_env=True)

        self.obs = self.env.reset()
        self.state_dim = self.obs_to_compact_state(self.obs, self.env.width, self.env.height, self.state_object_balls).shape[0]

        self.cfg = self.PPOConfig()
        self.cfg.lr = 3e-4 if self._guided_training_enabled() else 1e-4
        self.cfg.epochs = 6
        self.cfg.batch_size = 64 if self._guided_training_enabled() else 128
        self.cfg.rollout_steps = 256 if self._guided_training_enabled() else 512
        self.cfg.entropy_coef = 0.01 if self._guided_training_enabled() else 0.005
        self.agent = self.PPO_Agent(self.state_dim, self.cfg)

        self.ep = 0
        self.games = 0
        self.wins = 0
        self.history = {"episode_reward": [], "win_rate": [], "eval_win_rate": []}
        self.global_step = 0
        self.ep_update_stats = []
        self.training_progress_made = False

        self.ep_reward = 0.0
        self.ep_hits = 0
        self.ep_pots = 0
        self.total_hits = 0
        self.total_pots = 0
        self.done = False
        self.paused = False
        self.guided_warmup_episodes = 1200
        self.guided_warmup_prob = 1.0
        self.guided_assist_min_prob = 0.35
        self.guided_assist_decay_episodes = 1800
        self.guided_warmup_used = 0
        self.guided_imitation_loss = 0.0

        self.buffer = {"states": [], "actions_tanh": [], "logp": [], "values": [], "rewards": [], "dones": []}
        self.eval_interval = 20
        self.eval_episodes = 12
        self.best_eval_win_rate = 0.0
        self.last_eval_win_rate = 0.0

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
                    "layout": self.layout,
                    "workout_plan": self.workout_plan,
                    "target_ball_count": self.target_ball_count,
                    "target_random_balls": self.target_random_balls,
                    "target_layout": self.target_layout,
                    "target_opponent_type": self.target_opponent_type,
                    "state_object_balls": self.state_object_balls,
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

    def _normalize_layout(self, layout):
        layout = str(layout or "rack").lower()
        if layout in ("beginner", "easy"):
            return "beginner"
        return "rack"

    def _target_stage(self, name, start_ep):
        return {
            "name": name,
            "start_ep": int(start_ep),
            "ball_count": self.target_ball_count,
            "random_balls": self.target_random_balls,
            "layout": self.target_layout,
            "opponent_type": self.target_opponent_type,
        }

    def _build_workout_plan(self):
        stage_count = 11 if self.target_random_balls else 10
        if self.total_episodes < 1500:
            step = max(10, self.total_episodes // stage_count)
            starts = [step * i for i in range(stage_count)]
        else:
            starts = [0, 100, 220, 360, 540, 760, 1000, 1240, 1480, 1700, 1900]

        stages = [
            {
                "name": "1. Aim Warmup",
                "start_ep": starts[0],
                "ball_count": 2,
                "random_balls": False,
                "layout": "beginner",
                "opponent_type": "none",
            },
            {
                "name": "2. Three-Ball Drill",
                "start_ep": starts[1],
                "ball_count": 3,
                "random_balls": False,
                "layout": "beginner",
                "opponent_type": "none",
            },
            {
                "name": "3. Random Position Drill",
                "start_ep": starts[2],
                "ball_count": 3,
                "random_balls": True,
                "layout": "beginner",
                "opponent_type": "none",
            },
            {
                "name": "4. Five-Ball Solo",
                "start_ep": starts[3],
                "ball_count": 5,
                "random_balls": True,
                "layout": "beginner",
                "opponent_type": "none",
            },
            {
                "name": "5. Beginner vs Random",
                "start_ep": starts[4],
                "ball_count": 5,
                "random_balls": True,
                "layout": "beginner",
                "opponent_type": "random",
            },
            {
                "name": "6. Seven-Ball Beginner",
                "start_ep": starts[5],
                "ball_count": 7,
                "random_balls": True,
                "layout": "beginner",
                "opponent_type": "random",
            },
            {
                "name": "7. Small Rack",
                "start_ep": starts[6],
                "ball_count": 7,
                "random_balls": False,
                "layout": "rack",
                "opponent_type": "random",
            },
            {
                "name": "8. Medium Rack",
                "start_ep": starts[7],
                "ball_count": 10,
                "random_balls": False,
                "layout": "rack",
                "opponent_type": "random",
            },
            {
                "name": "9. Near-Full Rack",
                "start_ep": starts[8],
                "ball_count": 12,
                "random_balls": False,
                "layout": "rack",
                "opponent_type": "random",
            },
        ]

        if self.target_random_balls:
            stages.append({
                "name": "10. Random Full Rack Prep",
                "start_ep": starts[9],
                "ball_count": 15,
                "random_balls": True,
                "layout": "rack",
                "opponent_type": "random",
            })
            stages.append(self._target_stage("11. Target Game", starts[10]))
        else:
            stages.append(self._target_stage("10. Target Game", starts[9]))

        compact = []
        for stage in stages:
            if compact and all(compact[-1][key] == stage[key] for key in ("ball_count", "random_balls", "layout", "opponent_type")):
                continue
            compact.append(stage)
        return compact

    def _apply_curriculum_stage(self, index, create_env=False):
        index = max(0, min(index, len(self.curriculum) - 1))
        stage = self.curriculum[index]
        self.curriculum_stage = index
        self.max_ball_count = int(stage["ball_count"])
        self.random_balls = bool(stage["random_balls"])
        self.layout = self._normalize_layout(stage["layout"])
        self.opponent_type = stage["opponent_type"]

        if create_env:
            self.env = self.PoolEnvironment(
                render_mode="rgb_array",
                ball_count=self.max_ball_count,
                random_balls=self.random_balls,
                layout=self.layout,
            )
            return

        if self.env is not None:
            self.env.close()
        self.env = self.PoolEnvironment(
            render_mode="rgb_array",
            ball_count=self.max_ball_count,
            random_balls=self.random_balls,
            layout=self.layout,
        )
        self.obs = self.env.reset()
        self.ep_reward = 0.0
        self.ep_hits = 0
        self.ep_pots = 0
        self.done = False
        self.ep_update_stats = []
        self.buffer = {"states": [], "actions_tanh": [], "logp": [], "values": [], "rewards": [], "dones": []}

    def _maybe_advance_curriculum(self):
        if not self.workout_plan or self.curriculum_stage >= len(self.curriculum) - 1:
            return
        next_stage = self.curriculum[self.curriculum_stage + 1]
        if self.ep >= int(next_stage["start_ep"]):
            self._apply_curriculum_stage(self.curriculum_stage + 1)

    def _beginner_drill_enabled(self):
        return self.layout == "beginner" and self.max_ball_count == 2 and self.opponent_type == "none"

    def _guided_training_enabled(self):
        return self.layout == "beginner"

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
            "hits": self.total_hits,
            "pots": self.total_pots,
            "best_eval_win_rate": self.best_eval_win_rate,
            "ball_count": self.target_ball_count,
            "layout": self.target_layout,
            "random_balls": self.target_random_balls,
            "opponent_type": self.target_opponent_type,
            "workout_plan": self.workout_plan,
            "curriculum_stage": self.curriculum_stage,
            "state_object_balls": self.state_object_balls,
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
        if not self.resume_checkpoint:
            print("Starting a fresh training session without loading checkpoint weights.")
            return
        if not os.path.exists(self.checkpoint_path):
            return
        try:
            payload = self.torch.load(self.checkpoint_path, map_location=self.agent.device)
            model_state = payload.get("model")
            if not isinstance(model_state, dict) or not self._torch_state_is_finite(model_state):
                print(f"Skipping invalid checkpoint model: {self.checkpoint_path}")
                return

            first_layer = model_state.get("shared.0.weight")
            if first_layer is not None and int(first_layer.shape[1]) != self.state_dim:
                print("Skipping checkpoint with old state size. Start a new training run.")
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
                self.total_hits = int(payload.get("hits", self.total_hits))
                self.total_pots = int(payload.get("pots", self.total_pots))
                if self.workout_plan:
                    loaded_stage = int(payload.get("curriculum_stage", self.curriculum_stage))
                    self._apply_curriculum_stage(loaded_stage)
        except Exception as exc:
            print(f"Checkpoint load skipped: {exc}")

    def _random_action(self):
        angle = random.uniform(0, 2 * math.pi)
        power = random.uniform(0, 25)
        return {"angle": angle, "power": power}

    def _guided_beginner_action(self):
        return guided_shot_from_observation(
            self.env._get_observation(),
            self.env.width,
            self.env.height,
            self.env.holes,
            jitter=0.0,
        )

    def _guided_action_probability(self):
        if not self._guided_training_enabled():
            return 0.0
        if self.ep < self.guided_warmup_episodes:
            return self.guided_warmup_prob

        elapsed = self.ep - self.guided_warmup_episodes
        decay = math.exp(-elapsed / max(1.0, self.guided_assist_decay_episodes))
        return self.guided_assist_min_prob + (self.guided_warmup_prob - self.guided_assist_min_prob) * decay

    def _select_train_action(self, s_vec):
        guide_prob = self._guided_action_probability()
        if guide_prob > 0.0 and random.random() < guide_prob:
            action = self._guided_beginner_action()
            a_tanh = self.agent.action_to_tanh(action)
            self.guided_imitation_loss = self.agent.imitation_update(s_vec, a_tanh)
            logp, val = self.agent.logp_value_for_action(s_vec, a_tanh)
            self.guided_warmup_used += 1
            return action, logp, val, a_tanh
        return self.agent.select_action(s_vec)

    def _finish_episode(self):
        self.games += 1
        win = 1 if self.env.winner == 0 else 0
        if win:
            self.wins += 1

        self.history["episode_reward"].append(self.ep_reward)
        self.history["win_rate"].append(self.wins / self.games)
        self.ep += 1

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
                "episode_hits": self.ep_hits,
                "episode_pots": self.ep_pots,
                "total_hits": self.total_hits,
                "total_pots": self.total_pots,
                "num_wins": self.wins,
                "num_matches": self.games,
                "win_rate": win_rate,
                "win_percentage": win_percentage,
                "eval_win_rate": self.last_eval_win_rate,
                "best_eval_win_rate": self.best_eval_win_rate,
                "game_type": self._game_type_label(),
                "number_of_balls": self.max_ball_count,
                "target_number_of_balls": self.target_ball_count,
                "player1_score": p1_score,
                "player2_score": p2_score,
                "player1_remaining_balls": p1_remaining,
                "player2_remaining_balls": p2_remaining,
                "winner": winner,
                "random_balls": int(self.random_balls),
                "layout": self.layout,
                "opponent_type": self.opponent_type,
                "workout_plan": int(self.workout_plan),
                "curriculum_stage": self.curriculum_stage + 1,
                "guided_action_count": self.guided_warmup_used,
                "guided_imitation_loss": self.guided_imitation_loss,
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

        self._maybe_advance_curriculum()

    def _reset_episode(self):
        self.obs = self.env.reset()
        self.ep_reward = 0.0
        self.ep_hits = 0
        self.ep_pots = 0
        self.done = False
        self.ep_update_stats = []

    def _evaluate_policy(self, episodes):
        eval_env = self.PoolEnvironment(
            render_mode=None,
            ball_count=self.max_ball_count,
            random_balls=self.random_balls,
            layout=self.layout
        )
        wins = 0
        max_turns = 80
        for _ in range(episodes):
            obs = eval_env.reset()
            done = False
            turns = 0
            while not done and turns < max_turns:
                s_vec = self.obs_to_compact_state(obs, eval_env.width, eval_env.height, self.state_object_balls)
                if eval_env.current_player == 0:
                    action, _, _, _ = self.agent.select_action(s_vec, deterministic=True)
                    if self._guided_training_enabled():
                        guided_action = guided_shot_from_observation(
                            obs,
                            eval_env.width,
                            eval_env.height,
                            eval_env.holes,
                        )
                        guided_tanh = self.agent.action_to_tanh(guided_action)
                        policy_tanh = self.agent.action_to_tanh(action)
                        if np.linalg.norm(policy_tanh - guided_tanh) > 0.35:
                            action = guided_action
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

        s_vec = self.obs_to_compact_state(self.obs, self.env.width, self.env.height, self.state_object_balls)

        if self.env.current_player == 0:
            action, logp, val, a_tanh = self._select_train_action(s_vec)
            next_obs, reward, done, info = self.env.step(action)

            if info.get("first_ball_hit") is not None:
                self.ep_hits += 1
                self.total_hits += 1
            potted_count = int(info.get("potted_count", 0))
            self.ep_pots += potted_count
            self.total_pots += potted_count

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
            bootstrap_value = 0.0 if done else self.agent.value(
                self.obs_to_compact_state(self.obs, self.env.width, self.env.height, self.state_object_balls)
            )
            values = np.array(self.buffer["values"] + [bootstrap_value], dtype=np.float32)
            rewards = np.array(self.buffer["rewards"], dtype=np.float32)
            rewards = self.agent.normalize_rewards(rewards)
            dones = np.array(self.buffer["dones"], dtype=np.float32)
            adv, ret = self.agent._gae(rewards, values, dones)
            self.buffer["adv"] = adv.tolist()
            self.buffer["returns"] = ret.tolist()

            stats = self.agent.update(self.buffer)
            self.training_progress_made = True
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
        avg_hits = 0.0 if self.games == 0 else self.total_hits / self.games
        avg_pots = 0.0 if self.games == 0 else self.total_pots / self.games
        guide_prob = self._guided_action_probability()
        if self._guided_training_enabled() and self.ep < self.guided_warmup_episodes:
            guidance = f"Warmup {guide_prob:.0%}"
        elif guide_prob > 0:
            guidance = f"Assist {guide_prob:.0%}"
        else:
            guidance = "Off"
        if self.workout_plan:
            stage = self.curriculum[self.curriculum_stage]
            plan_line = (
                f"Workout: {self.curriculum_stage + 1}/{len(self.curriculum)} {stage['name']} "
                f"| Target: {self.target_ball_count} balls, {self.target_layout.title()}, "
                f"{'Random Positions' if self.target_random_balls else 'Fixed Positions'}, {self.target_opponent_type}"
            )
        else:
            plan_line = "Workout: Off"
        lines = [
            f"Episode: {self.ep}/{self.total_episodes} | Status: {status}",
            f"Episode Reward: {self.ep_reward:.2f} | Win Rate: {win_rate:.2f}",
            f"Hits/Game: {avg_hits:.2f} | Pots/Game: {avg_pots:.2f} | Guidance: {guidance}",
            f"Guided Shots: {self.guided_warmup_used} | Imitation Loss: {self.guided_imitation_loss:.4f}",
            f"Eval Win Rate: {self.last_eval_win_rate:.2f} | Best: {self.best_eval_win_rate:.2f}",
            plan_line,
            f"Balls: {self.max_ball_count} | Layout: {self.layout.title()} | Random Positions: {'On' if self.random_balls else 'Off'}",
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
            if self.training_progress_made or self.ep > 0:
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
