import os
import math
import copy

from checkpoint_utils import (
    LATEST_CHECKPOINT,
    checkpoint_quality_score,
    checkpoint_quality_too_low,
    checkpoint_search_paths,
)
from game_config import HEIGHT, WIDTH
from shot_guidance import action_has_target_contact, guided_shot_from_observation


class TrainedAgentPlayer:
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path or os.path.join("checkpoints", LATEST_CHECKPOINT)
        self.ready = False
        self.error = None
        self.using_policy = False
        self.ball_count = 15
        self.state_object_balls = 15
        self.state_dim = None
        self.agent = None

        try:
            import torch
            from ppo import PPOConfig, PPO_Agent, obs_to_compact_state
        except Exception as exc:
            self.error = f"Using built-in AI: {exc}"
            self.ready = True
            return

        self.torch = torch
        self.obs_to_compact_state = obs_to_compact_state

        checkpoint = self._load_best_checkpoint(checkpoint_path)
        if checkpoint is None:
            self.error = self.error or "Using built-in AI: no usable checkpoint found."
            self.ready = True
            return
        self.checkpoint_path, payload, model_state = checkpoint

        try:
            first_layer = model_state.get("shared.0.weight")
            if first_layer is None:
                self.error = "Using built-in AI: could not read model input size from checkpoint."
                self.ready = True
                return

            self.state_dim = int(first_layer.shape[1])
            self.state_object_balls = self._ball_count_from_state_dim(self.state_dim)
            self.ball_count = self._payload_ball_count(payload, self.state_object_balls)
            if checkpoint_quality_too_low(payload):
                self.error = "Using shot planner: checkpoint quality is too low."
                self.ready = True
                self.using_policy = False
                return

            cfg = PPOConfig()
            self.agent = PPO_Agent(self.state_dim, cfg)
            self.agent.net.load_state_dict(model_state)
            self.ready = True
            self.using_policy = True
        except Exception as exc:
            self.error = f"Using built-in AI: {exc}"
            self.ready = True

    def _load_best_checkpoint(self, checkpoint_path):
        best = None
        best_score = None
        errors = []

        for path in checkpoint_search_paths(checkpoint_path):
            if not os.path.exists(path):
                continue
            try:
                payload = self.torch.load(path, map_location="cpu")
                model_state = payload.get("model") if isinstance(payload, dict) else None
                if not isinstance(model_state, dict):
                    errors.append(f"{path}: no model")
                    continue
                if not self._state_is_finite(model_state):
                    errors.append(f"{path}: invalid model values")
                    continue

                score = checkpoint_quality_score(payload)
                if best is None or score > best_score:
                    best = (path, payload, model_state)
                    best_score = score
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        if best is None and errors:
            self.error = "Using built-in AI: " + errors[0]
        return best

    def _state_is_finite(self, value):
        if self.torch.is_tensor(value):
            if value.is_floating_point() or value.is_complex():
                return bool(self.torch.isfinite(value).all().item())
            return True
        if isinstance(value, dict):
            return all(self._state_is_finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(self._state_is_finite(v) for v in value)
        return True

    def _ball_count_from_state_dim(self, state_dim):
        # obs_to_compact_state stores 6 values per ball, including the cue ball,
        # and 12 extra game-state values.
        ball_count = (state_dim - 18) // 6
        if ball_count < 1 or ball_count > 15 or (state_dim - 18) % 6 != 0:
            return 15
        return ball_count

    def _payload_ball_count(self, payload, fallback):
        try:
            count = int(payload.get("ball_count", fallback))
        except (TypeError, ValueError):
            return fallback
        return max(2, min(15, count))

    def choose_action(self, game):
        obs = self._game_to_observation(game)
        fallback = self._guided_action(obs, game)
        planned = self._lookahead_action(obs, game, fallback)
        if planned is not None:
            fallback = planned
        if not self.using_policy or self.agent is None:
            return fallback

        state_vec = self.obs_to_compact_state(obs, WIDTH, HEIGHT, max_object_balls=self.state_object_balls)
        if state_vec.shape[0] != self.state_dim:
            return fallback
        action, _, _, _ = self.agent.select_action(state_vec, deterministic=True)
        return self._stabilize_action(obs, action, fallback)

    def _guided_action(self, obs, game):
        if not obs.get("ball_assignment_done"):
            break_action = self._break_action(obs)
            if break_action:
                return break_action

        holes = getattr(game, "holes", None)
        ball_radius = float(getattr(getattr(game, "cue_ball", None), "radius", 12.0))
        return guided_shot_from_observation(obs, WIDTH, HEIGHT, holes, ball_radius=ball_radius)

    def _break_action(self, obs):
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
        if rack_x < WIDTH * 0.55 or spread_x > 220 or spread_y > 180:
            return None

        return {
            "angle": math.atan2(rack_y - cue["y"], rack_x - cue["x"]) % (2 * math.pi),
            "power": 22.0,
        }

    def _stabilize_action(self, obs, action, fallback):
        power = float(action.get("power", 0.0))
        if 4.0 <= power <= 22.0 and action_has_target_contact(obs, action):
            model_angle = float(action.get("angle", 0.0))
            fallback_angle = float(fallback.get("angle", 0.0))
            angle_gap = abs((model_angle - fallback_angle + math.pi) % (2 * math.pi) - math.pi)
            fallback_power = float(fallback.get("power", 0.0))
            if angle_gap <= 0.18 and abs(power - fallback_power) <= 3.0:
                self.error = None
                return action

        if self.using_policy:
            self.error = "Using guided AI shot"
        return fallback

    def _lookahead_action(self, obs, game, fallback):
        try:
            from enviorment import PoolEnvironment
        except Exception:
            return None

        candidates = self._candidate_actions(obs, game, fallback)
        if not candidates:
            return None

        acting_player = int(obs.get("current_player", 0))
        before_potted = {
            int(ball.get("number", -1))
            for ball in obs.get("balls", [])
            if ball.get("potted", False)
        }
        best_action = None
        best_score = None
        for action in candidates:
            env = self._copy_game_to_env(game, PoolEnvironment)
            if env is None:
                return None
            _, reward, _, info = env.step(action)
            score = self._score_simulated_shot(env, reward, info, acting_player, before_potted)
            env.close()
            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        if best_action is not None and best_score is not None and best_score > -8.0:
            return best_action
        return None

    def _candidate_actions(self, obs, game, fallback):
        candidates = []
        ball_radius = float(getattr(getattr(game, "cue_ball", None), "radius", 14.0))

        def add(angle, power):
            candidates.append({
                "angle": float(angle) % (2 * math.pi),
                "power": max(0.0, min(25.0, float(power))),
            })

        add(fallback.get("angle", 0.0), fallback.get("power", 8.0))
        for delta in (-0.24, -0.16, -0.08, 0.08, 0.16, 0.24):
            add(float(fallback.get("angle", 0.0)) + delta, fallback.get("power", 8.0))
        for power in (5.5, 7.5, 10.0, 13.0, 17.0, 22.0):
            add(fallback.get("angle", 0.0), power)

        cue = None
        object_balls = []
        for ball in obs.get("balls", []):
            if ball.get("potted", False):
                continue
            if ball.get("number", 0) == 0:
                cue = ball
            else:
                object_balls.append(ball)

        if cue and object_balls:
            if self._should_include_break_candidates(obs, object_balls):
                rack_x = sum(ball["x"] for ball in object_balls) / len(object_balls)
                rack_y = sum(ball["y"] for ball in object_balls) / len(object_balls)
                break_angle = math.atan2(rack_y - cue["y"], rack_x - cue["x"])
                for delta in (-0.08, -0.04, 0.0, 0.04, 0.08):
                    add(break_angle + delta, 22.0)

            holes = getattr(game, "holes", None) or [(80, 80), (600, 70), (1120, 80), (80, 720), (600, 730), (1120, 720)]
            for target in object_balls:
                if not self._is_legal_target(obs, target):
                    continue
                center_angle = math.atan2(target["y"] - cue["y"], target["x"] - cue["x"])
                distance = math.hypot(target["x"] - cue["x"], target["y"] - cue["y"])
                add(center_angle, max(5.0, min(18.0, distance / 70.0)))
                for hole in holes:
                    dx = hole[0] - target["x"]
                    dy = hole[1] - target["y"]
                    dist = math.hypot(dx, dy)
                    if dist <= 1e-6:
                        continue
                    ghost_x = target["x"] - dx / dist * (ball_radius * 2.0)
                    ghost_y = target["y"] - dy / dist * (ball_radius * 2.0)
                    angle = math.atan2(ghost_y - cue["y"], ghost_x - cue["x"])
                    power = max(5.0, min(20.0, (math.hypot(ghost_x - cue["x"], ghost_y - cue["y"]) + dist) / 85.0))
                    add(angle, power)

        unique = {}
        for action in candidates:
            key = (round(action["angle"], 3), round(action["power"], 2))
            unique[key] = action
        return list(unique.values())[:80]

    def _should_include_break_candidates(self, obs, object_balls):
        if obs.get("ball_assignment_done"):
            return False
        if len(object_balls) < 3:
            return False

        rack_x = sum(ball["x"] for ball in object_balls) / len(object_balls)
        spread_x = max(ball["x"] for ball in object_balls) - min(ball["x"] for ball in object_balls)
        spread_y = max(ball["y"] for ball in object_balls) - min(ball["y"] for ball in object_balls)
        return rack_x >= WIDTH * 0.55 and spread_x <= 220 and spread_y <= 180

    def _copy_game_to_env(self, game, PoolEnvironment):
        try:
            ball_count = max(2, min(15, len(getattr(game, "balls", []))))
            env = PoolEnvironment(render_mode=None, ball_count=ball_count, random_balls=False, layout="rack")
            env.ball_radius = float(getattr(game.cue_ball, "radius", env.ball_radius))
            env.hole_radius = 35
            env.cue_ball.x = float(game.cue_ball.x)
            env.cue_ball.y = float(game.cue_ball.y)
            env.cue_ball.vx = float(getattr(game.cue_ball, "vx", 0.0))
            env.cue_ball.vy = float(getattr(game.cue_ball, "vy", 0.0))
            env.cue_ball.potted = bool(getattr(game.cue_ball, "potted", False))
            env.cue_ball.radius = env.ball_radius

            for dst, src in zip(env.balls, game.balls):
                dst.x = float(src.x)
                dst.y = float(src.y)
                dst.vx = float(getattr(src, "vx", 0.0))
                dst.vy = float(getattr(src, "vy", 0.0))
                dst.potted = bool(getattr(src, "potted", False))
                dst.number = int(getattr(src, "number", dst.number))
                dst.is_striped = bool(getattr(src, "is_striped", dst.is_striped))
                dst.radius = float(getattr(src, "radius", env.ball_radius))

            env.players = copy.deepcopy(getattr(game, "players", env.players))
            env.current_player = int(getattr(game, "current_player", 0))
            env.turn_ended = bool(getattr(game, "turn_ended", False))
            env.foul = bool(getattr(game, "foul", False))
            env.ball_assignment_done = bool(getattr(game, "ball_assignment_done", False))
            env.first_ball_hit = None
            env.winner = getattr(game, "winner", None)
            env.shots_taken = int(getattr(game, "shots_taken", 0))
            return env
        except Exception:
            return None

    def _score_simulated_shot(self, env, reward, info, acting_player, before_potted):
        score = float(reward)
        if env.winner == acting_player:
            score += 100.0
        elif env.winner is not None:
            score -= 100.0

        potted = [
            ball for ball in env.balls
            if ball.potted and int(ball.number) not in before_potted
        ]
        object_pots = [ball for ball in potted if ball.number != 8]
        eight_potted = any(ball.number == 8 for ball in potted)
        own_pots = 0
        wrong_pots = 0
        player_type = None
        if 0 <= acting_player < len(env.players):
            player_type = env.players[acting_player].get("type")
        for ball in object_pots:
            ball_type = "striped" if ball.is_striped else "solid"
            if player_type is None or ball_type == player_type:
                own_pots += 1
            else:
                wrong_pots += 1

        score += 18.0 * own_pots
        score -= 10.0 * wrong_pots
        if eight_potted and env.winner != acting_player:
            score -= 60.0
        if info.get("first_ball_hit") is None:
            score -= 80.0
        if bool(getattr(env, "foul", False)):
            score -= 10.0
        return score

    def _should_play_eight(self, obs):
        player_types = obs.get("player_types", [None, None])
        current_player = int(obs.get("current_player", 0))
        player_type = player_types[current_player] if current_player < len(player_types) else None
        if player_type not in ("solid", "striped"):
            return False
        for ball in obs.get("balls", []):
            if ball.get("potted", False) or ball.get("number", 0) in (0, 8):
                continue
            ball_type = "striped" if ball.get("is_striped", False) else "solid"
            if ball_type == player_type:
                return False
        return True

    def _is_legal_target(self, obs, ball):
        number = int(ball.get("number", 0))
        if number == 8:
            return self._should_play_eight(obs)
        if number == 0:
            return False

        player_types = obs.get("player_types", [None, None])
        current_player = int(obs.get("current_player", 0))
        player_type = player_types[current_player] if current_player < len(player_types) else None
        if player_type not in ("solid", "striped"):
            return True

        ball_type = "striped" if ball.get("is_striped", False) else "solid"
        return ball_type == player_type

    def _game_to_observation(self, game):
        balls = [self._ball_state(game.cue_ball)]
        balls.extend(self._ball_state(ball) for ball in game.balls)
        return {
            "balls": balls,
            "current_player": game.current_player,
            "player_types": [game.players[0]["type"], game.players[1]["type"]],
            "ball_assignment_done": game.ball_assignment_done,
            "foul": game.foul,
            "turn_ended": game.turn_ended,
            "winner": game.winner,
        }

    def _ball_state(self, ball):
        return {
            "x": ball.x,
            "y": ball.y,
            "vx": ball.vx,
            "vy": ball.vy,
            "potted": ball.potted,
            "number": ball.number,
            "is_striped": ball.is_striped,
            "radius": getattr(ball, "radius", 12),
        }
