import math
import os
import random

import numpy as np
import wandb

from checkpoint_utils import (
    BEST_CHECKPOINT,
    LATEST_CHECKPOINT,
    canonical_checkpoint_path,
    checkpoint_quality_score,
    checkpoint_search_paths,
)
from shot_guidance import action_has_target_contact, guided_shot_from_observation

class TrainingController:
    def __init__(self, total_episodes=8000, ball_count=15, random_balls=False, opponent_type="random", resume_checkpoint=False, layout="rack", workout_plan=False):
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
        self.curriculum_stage_start_ep = 0
        self.stage_recent_wins = []
        self.stage_recent_pots = []
        self.stage_games = 0
        self.stage_wins = 0
        self.stage_pots = 0
        self.stage_eval_win_rate = 0.0
        self.stage_best_eval_win_rate = 0.0
        self.best_curriculum_score = -1.0
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
        self._configure_learning_for_stage()

        self.ep = 0
        self.games = 0
        self.wins = 0
        self.history = {"episode_reward": [], "win_rate": [], "lifetime_win_rate": [], "eval_win_rate": []}
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
        self.guided_warmup_episodes = 0
        self.guided_warmup_prob = 0.0
        self.guided_assist_min_prob = 0.0
        self.guided_assist_decay_episodes = 1
        self.guided_warmup_used = 0
        self.guided_imitation_loss = 0.0
        self.guided_imitation_steps = 8
        self.guided_replay_batch = 16
        self.guided_replay_limit = 2048
        self.guided_replay = []

        self.buffer = {"states": [], "actions_tanh": [], "logp": [], "values": [], "rewards": [], "dones": []}
        self.eval_interval = 20
        self.eval_episodes = 12
        self.best_eval_win_rate = 0.0
        self.last_eval_win_rate = 0.0

        self.checkpoint_path = canonical_checkpoint_path(LATEST_CHECKPOINT)
        self.best_checkpoint_path = canonical_checkpoint_path(BEST_CHECKPOINT)
        self._load_checkpoint_if_exists()
        self.wandb_run = self._init_wandb_run()

        self.ready = True

    def _configure_learning_for_stage(self):
        if not hasattr(self, "cfg") or not hasattr(self, "agent"):
            return

        if self.layout == "beginner":
            self.cfg.lr = 2e-4
            self.cfg.batch_size = 64
            self.cfg.rollout_steps = 256
            self.cfg.entropy_coef = 0.01
        else:
            self.cfg.lr = 1e-4
            self.cfg.batch_size = 128
            self.cfg.rollout_steps = 512
            self.cfg.entropy_coef = 0.006

        for group in self.agent.opt.param_groups:
            group["lr"] = self.cfg.lr

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

    def _make_stage(
        self,
        name,
        ball_count,
        random_balls,
        layout,
        opponent_type,
        min_episodes,
        max_episodes,
        mastery_win_rate,
        mastery_pots=None,
        guide_start=0.0,
        guide_end=0.0,
        advance_floor=0.0,
        regression_floor=0.0,
    ):
        return {
            "name": name,
            "start_ep": 0,
            "ball_count": max(2, min(15, int(ball_count))),
            "random_balls": bool(random_balls),
            "layout": self._normalize_layout(layout),
            "opponent_type": opponent_type,
            "min_episodes": int(min_episodes),
            "max_episodes": int(max_episodes),
            "mastery_win_rate": float(mastery_win_rate),
            "mastery_pots": None if mastery_pots is None else float(mastery_pots),
            "guide_start": float(guide_start),
            "guide_end": float(guide_end),
            "advance_floor": float(advance_floor),
            "regression_floor": float(regression_floor),
        }

    def _target_stage(self, name, start_ep):
        beginner_guidance = self.target_layout == "beginner"
        small_rack_guidance = self.target_layout == "rack" and self.target_ball_count <= 6
        return {
            "name": name,
            "start_ep": int(start_ep),
            "ball_count": self.target_ball_count,
            "random_balls": self.target_random_balls,
            "layout": self.target_layout,
            "opponent_type": self.target_opponent_type,
            "min_episodes": max(1, self.total_episodes),
            "max_episodes": max(1, self.total_episodes),
            "mastery_win_rate": 1.0,
            "mastery_pots": None,
            "guide_start": 1.0 if beginner_guidance else (0.65 if small_rack_guidance else 0.0),
            "guide_end": 1.0 if beginner_guidance else (0.15 if small_rack_guidance else 0.0),
            "advance_floor": 0.0,
            "regression_floor": 0.0,
        }

    def _stage_starts(self, stage_count):
        if stage_count <= 1:
            return [0]
        step = max(8, self.total_episodes // stage_count)
        return [step * i for i in range(stage_count)]

    def _build_workout_plan(self):
        stage_budget = max(120, self.total_episodes // 26)
        short_min = max(30, int(stage_budget * 0.22))
        normal_min = max(50, int(stage_budget * 0.32))
        long_min = max(70, int(stage_budget * 0.42))
        short_max = max(120, int(stage_budget * 0.85))
        normal_max = max(180, int(stage_budget * 1.15))
        long_max = max(260, int(stage_budget * 1.50))

        def beginner_stage(name, balls, random_positions, opponent, min_ep, max_ep, win_rate, pots, guide_start, guide_end):
            return self._make_stage(
                name,
                balls,
                random_positions,
                "beginner",
                opponent,
                min_ep,
                max_ep,
                win_rate,
                pots,
                guide_start,
                guide_end,
                advance_floor=max(0.35, win_rate - 0.18),
                regression_floor=max(0.20, win_rate - 0.35),
            )

        def rack_stage(name, balls, random_positions, opponent, min_ep, max_ep, win_rate, pots=None, guide_start=0.30, guide_end=0.06):
            return self._make_stage(
                name,
                balls,
                random_positions,
                "rack",
                opponent,
                min_ep,
                max_ep,
                win_rate,
                pots,
                guide_start,
                guide_end,
                advance_floor=max(0.18, win_rate - 0.15),
                regression_floor=max(0.08, win_rate - 0.28),
            )

        stages = [
            beginner_stage("2-Ball Guided Aim", 2, False, "none", normal_min, normal_max, 0.85, 1.0, 0.95, 0.55),
            beginner_stage("2-Ball Policy Check", 2, False, "none", short_min, short_max, 0.85, 1.0, 0.30, 0.05),
            beginner_stage("3-Ball Guided Aim", 3, False, "none", normal_min, normal_max, 0.82, 1.0, 0.90, 0.45),
            beginner_stage("3-Ball Policy Check", 3, False, "none", short_min, short_max, 0.82, 1.0, 0.25, 0.05),
            beginner_stage("4-Ball Guided Aim", 4, False, "none", normal_min, normal_max, 0.78, 1.0, 0.85, 0.35),
            beginner_stage("4-Ball Policy Check", 4, False, "none", short_min, short_max, 0.78, 1.0, 0.20, 0.05),
            beginner_stage("4-Ball Random Intro", 4, True, "none", normal_min, normal_max, 0.72, 1.0, 0.75, 0.25),
            beginner_stage("4-Ball Random Check", 4, True, "none", short_min, short_max, 0.72, 1.0, 0.20, 0.05),
            beginner_stage("5-Ball Pattern Pots", 5, True, "none", long_min, long_max, 0.68, 2.0, 0.65, 0.20),
            beginner_stage("6-Ball Pattern Pots", 6, True, "none", long_min, long_max, 0.64, 2.0, 0.55, 0.15),
            beginner_stage("6-Ball Opponent Intro", 6, True, "random", long_min, long_max, 0.58, 2.0, 0.45, 0.10),
            beginner_stage("7-Ball Beginner Match", 7, True, "random", long_min, long_max, 0.54, 2.0, 0.35, 0.05),
            beginner_stage("8-Ball Beginner Match", 8, True, "random", long_min, long_max, 0.50, 2.0, 0.25, 0.05),
            rack_stage("2-Ball Rack Bridge", 2, False, "none", normal_min, long_max, 0.80, 1.0, 0.45, 0.12),
            rack_stage("3-Ball Rack Bridge", 3, False, "none", normal_min, long_max, 0.70, 1.0, 0.40, 0.10),
            rack_stage("4-Ball Rack Bridge", 4, False, "none", normal_min, long_max, 0.60, 1.0, 0.35, 0.08),
            rack_stage("5-Ball Rack Solo", 5, False, "none", normal_min, long_max, 0.50, 1.0),
            rack_stage("6-Ball Rack Solo", 6, False, "none", normal_min, long_max, 0.46, 1.0),
            rack_stage("7-Ball Rack Solo", 7, False, "none", normal_min, long_max, 0.42, 1.0),
            rack_stage("7-Ball Rack Opponent", 7, False, "random", normal_min, long_max, 0.38, 1.0),
            rack_stage("8-Ball Rack Opponent", 8, False, "random", normal_min, long_max, 0.35, 1.0),
            rack_stage("9-Ball Rack Opponent", 9, False, "random", normal_min, long_max, 0.32, 1.0),
            rack_stage("10-Ball Rack Opponent", 10, False, "random", normal_min, long_max, 0.30, 1.0),
            rack_stage("11-Ball Rack Opponent", 11, False, "random", normal_min, long_max, 0.28, 1.0),
            rack_stage("12-Ball Rack Opponent", 12, False, "random", normal_min, long_max, 0.26, 1.0),
            rack_stage("13-Ball Rack Opponent", 13, False, "random", normal_min, long_max, 0.24, 1.0),
            rack_stage("14-Ball Rack Opponent", 14, False, "random", normal_min, long_max, 0.22, 1.0),
            rack_stage("15-Ball Rack Prep", 15, False, "random", long_min, long_max, 0.20, 1.0),
        ]

        if self.target_random_balls:
            stages.append(rack_stage("15-Ball Random Rack Prep", 15, True, "random", long_min, long_max, 0.18, 1.0))

        stages.append(self._make_stage(
            "Target Game",
            self.target_ball_count,
            self.target_random_balls,
            self.target_layout,
            self.target_opponent_type,
            long_min,
            max(long_max, int(stage_budget * 2.0)),
            0.18 if self.target_layout == "rack" else 0.50,
            1.0,
            0.10 if self.target_layout == "beginner" else 0.0,
            0.0,
            0.0,
            0.0,
        ))

        starts = self._stage_starts(len(stages))
        for index, stage in enumerate(stages):
            stage["name"] = f"{index + 1}. {stage['name']}"
            stage["start_ep"] = starts[index]
        return stages

    def _apply_curriculum_stage(self, index, create_env=False):
        index = max(0, min(index, len(self.curriculum) - 1))
        stage = self.curriculum[index]
        self.curriculum_stage = index
        self.curriculum_stage_start_ep = self.ep if hasattr(self, "ep") else int(stage["start_ep"])
        self.stage_recent_wins = []
        self.stage_recent_pots = []
        self.stage_games = 0
        self.stage_wins = 0
        self.stage_pots = 0
        self.stage_eval_win_rate = 0.0
        self.stage_best_eval_win_rate = 0.0
        self.max_ball_count = int(stage["ball_count"])
        self.random_balls = bool(stage["random_balls"])
        self.layout = self._normalize_layout(stage["layout"])
        self.opponent_type = stage["opponent_type"]
        self._configure_learning_for_stage()

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

    def _remember_guided_example(self, state_vec, action_tanh):
        self.guided_replay.append((np.array(state_vec, dtype=np.float32), np.array(action_tanh, dtype=np.float32)))
        if len(self.guided_replay) > self.guided_replay_limit:
            self.guided_replay = self.guided_replay[-self.guided_replay_limit:]

    def _guided_imitation_update(self, state_vec, action_tanh):
        self._remember_guided_example(state_vec, action_tanh)
        losses = []
        for _ in range(self.guided_imitation_steps):
            losses.append(self.agent.imitation_update(state_vec, action_tanh))

        batch_size = min(self.guided_replay_batch, len(self.guided_replay))
        if batch_size > 0:
            for replay_state, replay_action in random.sample(self.guided_replay, batch_size):
                losses.append(self.agent.imitation_update(replay_state, replay_action))

        return float(np.mean(losses)) if losses else 0.0

    def _maybe_advance_curriculum(self):
        if not self.workout_plan or self.curriculum_stage >= len(self.curriculum) - 1:
            return
        if self._stage_mastered():
            self._apply_curriculum_stage(self.curriculum_stage + 1)

    def _beginner_drill_enabled(self):
        return self.layout == "beginner" and self.max_ball_count == 2 and self.opponent_type == "none"

    def _guided_training_enabled(self):
        stage = self._current_stage()
        return float(stage.get("guide_start", 0.0)) > 0.0 or float(stage.get("guide_end", 0.0)) > 0.0

    def _current_stage(self):
        if not self.curriculum:
            return {}
        return self.curriculum[self.curriculum_stage]

    def _episodes_in_stage(self):
        return max(0, self.ep - self.curriculum_stage_start_ep)

    def _stage_mastered(self):
        stage = self._current_stage()
        episodes_in_stage = self._episodes_in_stage()
        min_stage_episodes = int(stage.get("min_episodes", 0))
        max_stage_episodes = int(stage.get("max_episodes", max(1, min_stage_episodes)))
        required_win_rate = float(stage.get("mastery_win_rate", 0.5))
        advance_floor = float(stage.get("advance_floor", max(0.0, required_win_rate - 0.25)))

        if episodes_in_stage < min_stage_episodes:
            return False

        recent_games = min(20, len(self.stage_recent_wins))
        if recent_games == 0 or self.stage_eval_win_rate <= 0.0:
            return False

        recent_win_rate = sum(self.stage_recent_wins[-recent_games:]) / recent_games
        recent_pots = self.stage_recent_pots[-recent_games:]
        avg_pots = sum(recent_pots) / recent_games
        required_pots = stage.get("mastery_pots")
        pots_ready = required_pots is None or avg_pots >= float(required_pots)
        eval_ready = self.stage_eval_win_rate >= required_win_rate
        recent_ready = recent_win_rate >= max(advance_floor, required_win_rate - 0.10)

        if eval_ready and recent_ready and pots_ready:
            return True

        if episodes_in_stage >= max_stage_episodes:
            regression_floor = float(stage.get("regression_floor", max(0.0, advance_floor - 0.15)))
            if self.stage_eval_win_rate >= advance_floor and recent_win_rate >= regression_floor and pots_ready:
                return True

        return False

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
            "best_curriculum_score": self.best_curriculum_score,
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
            payload = self._checkpoint_payload()
            self.torch.save(payload, self.checkpoint_path)
            if best:
                existing = self._best_existing_payload()
                existing_score = checkpoint_quality_score(existing) if existing else -1.0
                new_score = checkpoint_quality_score(payload)
                if existing is None or new_score > existing_score + 1e-6:
                    self.torch.save(payload, self.best_checkpoint_path)
        except Exception:
            pass

    def _best_existing_payload(self):
        best_payload = None
        best_score = None
        for path in checkpoint_search_paths():
            if not os.path.exists(path):
                continue
            try:
                payload = self.torch.load(path, map_location=self.agent.device)
            except Exception:
                continue
            score = checkpoint_quality_score(payload)
            if best_payload is None or score > best_score:
                best_payload = payload
                best_score = score
        return best_payload

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
        checkpoint = self._find_best_compatible_checkpoint()
        if checkpoint is None:
            return
        path, payload, model_state = checkpoint
        try:
            self.agent.net.load_state_dict(model_state)

            optimizer_state = payload.get("optimizer")
            if self.resume_checkpoint and isinstance(optimizer_state, dict) and self._torch_state_is_finite(optimizer_state):
                self.agent.opt.load_state_dict(optimizer_state)
            elif self.resume_checkpoint:
                print("Skipping invalid checkpoint optimizer state.")

            if self.resume_checkpoint:
                for attr in ("reward_mean", "reward_var", "reward_count"):
                    value = float(payload.get(attr, getattr(self.agent, attr)))
                    if math.isfinite(value):
                        setattr(self.agent, attr, value)

            try:
                best_eval_win_rate = float(payload.get("best_eval_win_rate", self.best_eval_win_rate))
                if math.isfinite(best_eval_win_rate):
                    self.best_eval_win_rate = best_eval_win_rate
            except (TypeError, ValueError):
                pass

            try:
                best_curriculum_score = float(payload.get("best_curriculum_score", self.best_curriculum_score))
                if math.isfinite(best_curriculum_score):
                    self.best_curriculum_score = best_curriculum_score
            except (TypeError, ValueError):
                pass

            if self.resume_checkpoint:
                self.ep = int(payload.get("ep", self.ep))
                self.games = int(payload.get("games", self.games))
                self.wins = int(payload.get("wins", self.wins))
                self.total_hits = int(payload.get("hits", self.total_hits))
                self.total_pots = int(payload.get("pots", self.total_pots))
                if self.workout_plan:
                    loaded_stage = int(payload.get("curriculum_stage", self.curriculum_stage))
                    self._apply_curriculum_stage(loaded_stage)
            mode = "resumed" if self.resume_checkpoint else "warm-started"
            print(f"Training {mode} from checkpoint: {path}")
        except Exception as exc:
            print(f"Checkpoint load skipped: {exc}")

    def _find_best_compatible_checkpoint(self):
        best = None
        best_score = None
        for path in checkpoint_search_paths():
            if not os.path.exists(path):
                continue
            try:
                payload = self.torch.load(path, map_location=self.agent.device)
                model_state = payload.get("model") if isinstance(payload, dict) else None
                if not isinstance(model_state, dict) or not self._torch_state_is_finite(model_state):
                    print(f"Skipping invalid checkpoint model: {path}")
                    continue

                first_layer = model_state.get("shared.0.weight")
                if first_layer is not None and int(first_layer.shape[1]) != self.state_dim:
                    print(f"Skipping checkpoint with old state size: {path}")
                    continue

                score = checkpoint_quality_score(payload)
                if best is None or score > best_score:
                    best = (path, payload, model_state)
                    best_score = score
            except Exception as exc:
                print(f"Checkpoint load skipped for {path}: {exc}")
        return best

    def _random_action(self):
        angle = random.uniform(0, 2 * math.pi)
        power = random.uniform(0, 25)
        return {"angle": angle, "power": power}

    def _guided_beginner_action(self):
        obs = self.env._get_observation()
        break_action = self._guided_rack_break_action(obs)
        if break_action is not None:
            return break_action
        return guided_shot_from_observation(
            obs,
            self.env.width,
            self.env.height,
            self.env.holes,
            jitter=0.0,
        )

    def _guided_rack_break_action(self, obs):
        if self.layout != "rack" or obs.get("ball_assignment_done"):
            return None

        cue = None
        object_balls = []
        for ball in obs.get("balls", []):
            if ball.get("potted", False):
                continue
            if ball.get("number", 0) == 0:
                cue = ball
            else:
                object_balls.append(ball)

        if cue is None or len(object_balls) < 3:
            return None

        rack_x = sum(ball["x"] for ball in object_balls) / len(object_balls)
        rack_y = sum(ball["y"] for ball in object_balls) / len(object_balls)
        spread_x = max(ball["x"] for ball in object_balls) - min(ball["x"] for ball in object_balls)
        spread_y = max(ball["y"] for ball in object_balls) - min(ball["y"] for ball in object_balls)
        if rack_x < self.env.width * 0.55 or spread_x > 220 or spread_y > 180:
            return None

        return {
            "angle": math.atan2(rack_y - cue["y"], rack_x - cue["x"]) % (2 * math.pi),
            "power": 22.0,
        }

    def _guided_action_probability(self):
        if not self._guided_training_enabled():
            return 0.0
        stage = self._current_stage()
        guide_start = float(stage.get("guide_start", 0.0))
        guide_end = float(stage.get("guide_end", 0.0))
        if guide_start <= 0.0 and guide_end <= 0.0:
            return 0.0

        min_episodes = max(1, int(stage.get("min_episodes", 1)))
        progress = min(1.0, self._episodes_in_stage() / min_episodes)
        eased = progress * progress * (3.0 - 2.0 * progress)
        guide_prob = guide_start + (guide_end - guide_start) * eased
        return max(0.0, min(1.0, guide_prob))

    def _select_train_action(self, s_vec):
        guide_prob = self._guided_action_probability()
        if guide_prob > 0.0 and random.random() < guide_prob:
            action = self._guided_beginner_action()
            a_tanh = self.agent.action_to_tanh(action)
            self.guided_imitation_loss = self._guided_imitation_update(s_vec, a_tanh)
            logp, val = self.agent.logp_value_for_action(s_vec, a_tanh)
            self.guided_warmup_used += 1
            return action, logp, val, a_tanh

        action, logp, val, a_tanh = self.agent.select_action(s_vec)
        if self._guided_training_enabled():
            power = float(action.get("power", 0.0))
            if power < 4.0 or power > 22.0 or not action_has_target_contact(self.obs, action):
                action = self._guided_beginner_action()
                a_tanh = self.agent.action_to_tanh(action)
                self.guided_imitation_loss = self._guided_imitation_update(s_vec, a_tanh)
                logp, val = self.agent.logp_value_for_action(s_vec, a_tanh)
                self.guided_warmup_used += 1
        return action, logp, val, a_tanh

    def _finish_episode(self):
        self.games += 1
        win = 1 if self.env.winner == 0 else 0
        if win:
            self.wins += 1
        self.stage_games += 1
        if win:
            self.stage_wins += 1
        self.stage_pots += self.ep_pots

        self.history["episode_reward"].append(self.ep_reward)
        self.stage_recent_wins.append(win)
        self.stage_recent_pots.append(self.ep_pots)
        if len(self.stage_recent_wins) > 30:
            self.stage_recent_wins = self.stage_recent_wins[-30:]
            self.stage_recent_pots = self.stage_recent_pots[-30:]
        rolling_games = max(1, len(self.stage_recent_wins))
        rolling_win_rate = sum(self.stage_recent_wins) / rolling_games
        lifetime_win_rate = self.wins / self.games
        self.history["win_rate"].append(rolling_win_rate)
        self.history["lifetime_win_rate"].append(lifetime_win_rate)
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
        lifetime_win_rate = self.history["lifetime_win_rate"][-1]
        win_percentage = win_rate * 100.0
        stage_win_rate = 0.0 if self.stage_games == 0 else self.stage_wins / self.stage_games
        stage_avg_pots = 0.0 if self.stage_games == 0 else self.stage_pots / self.stage_games

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
                "rolling_win_rate": win_rate,
                "rolling_win_percentage": win_percentage,
                "lifetime_win_rate": lifetime_win_rate,
                "lifetime_win_percentage": lifetime_win_rate * 100.0,
                "stage_win_rate": stage_win_rate,
                "stage_avg_pots": stage_avg_pots,
                "stage_eval_win_rate": self.stage_eval_win_rate,
                "stage_best_eval_win_rate": self.stage_best_eval_win_rate,
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
            self.stage_eval_win_rate = self.last_eval_win_rate
            self.stage_best_eval_win_rate = max(self.stage_best_eval_win_rate, self.stage_eval_win_rate)
            self.history["eval_win_rate"].append(self.last_eval_win_rate)
            eval_score = (self.curriculum_stage + self.last_eval_win_rate) if self.workout_plan else self.last_eval_win_rate
            is_best = eval_score >= self.best_curriculum_score
            if is_best:
                self.best_curriculum_score = eval_score
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
        max_turns = max(20, int(eval_env.max_shots) + 10)
        for _ in range(episodes):
            obs = eval_env.reset()
            done = False
            turns = 0
            while not done and turns < max_turns:
                s_vec = self.obs_to_compact_state(obs, eval_env.width, eval_env.height, self.state_object_balls)
                if eval_env.current_player == 0:
                    action, _, _, _ = self.agent.select_action(s_vec, deterministic=True)
                    if self._guided_training_enabled():
                        power = float(action.get("power", 0.0))
                        if power < 4.0 or power > 22.0 or not action_has_target_contact(obs, action):
                            action = guided_shot_from_observation(obs, eval_env.width, eval_env.height, eval_env.holes)
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
            agent_reward = float(reward)

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
            self.buffer["rewards"].append(agent_reward)
            self.buffer["dones"].append(float(done))
        else:
            if self.opponent_type == "self":
                action, _, _, _ = self.agent.select_action(s_vec)
                next_obs, reward, done, _ = self.env.step(action)
            else:
                next_obs, reward, done, _ = self.env.step(self._random_action())
            agent_reward = -float(reward)
            if done and self.buffer["dones"]:
                self.buffer["dones"][-1] = 1.0
                self.buffer["rewards"][-1] += agent_reward

        self.obs = next_obs
        self.ep_reward += agent_reward
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

        lifetime_win_rate = 0.0 if self.games == 0 else self.wins / self.games
        win_rate = 0.0
        if self.stage_recent_wins:
            win_rate = sum(self.stage_recent_wins) / len(self.stage_recent_wins)
        stage_win_rate = 0.0 if self.stage_games == 0 else self.stage_wins / self.stage_games
        stage_avg_pots = 0.0 if self.stage_games == 0 else self.stage_pots / self.stage_games
        avg_hits = 0.0 if self.games == 0 else self.total_hits / self.games
        avg_pots = 0.0 if self.games == 0 else self.total_pots / self.games
        guide_prob = self._guided_action_probability()
        if guide_prob > 0:
            guidance = f"Assist {guide_prob:.0%}"
        else:
            guidance = "Off"
        if self.workout_plan:
            stage = self.curriculum[self.curriculum_stage]
            stage_prefix = f"{self.curriculum_stage + 1}. "
            stage_name = stage["name"]
            if stage_name.startswith(stage_prefix):
                stage_name = stage_name[len(stage_prefix):]
            plan_line = (
                f"Workout: {self.curriculum_stage + 1}/{len(self.curriculum)} {stage_name} "
                f"| Target: {self.target_ball_count} balls, {self.target_layout.title()}, "
                f"{'Random Positions' if self.target_random_balls else 'Fixed Positions'}, {self.target_opponent_type}"
            )
            stage_line = (
                f"Stage Episodes: {self._episodes_in_stage()}/{stage.get('min_episodes', 0)} min "
                f"| max {stage.get('max_episodes', 0)} | Need Eval {stage.get('mastery_win_rate', 0.0):.2f}"
            )
        else:
            plan_line = "Workout: Off"
            stage_line = "Stage: Selected settings"
        lines = [
            f"Episode: {self.ep}/{self.total_episodes} | Status: {status}",
            f"Episode Reward: {self.ep_reward:.2f} | Recent WR: {win_rate:.2f} | Lifetime WR: {lifetime_win_rate:.2f} | Stage WR: {stage_win_rate:.2f}",
            f"Hits/Game: {avg_hits:.2f} | Pots/Game: {avg_pots:.2f} | Stage Pots/Game: {stage_avg_pots:.2f}",
            f"Guided Shots: {self.guided_warmup_used} | Imitation Loss: {self.guided_imitation_loss:.4f}",
            f"Policy Eval: {self.stage_eval_win_rate:.2f} | Stage Best: {self.stage_best_eval_win_rate:.2f} | Guidance: {guidance}",
            plan_line,
            stage_line,
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
