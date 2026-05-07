import os
import math

from game_config import HEIGHT, WIDTH
from shot_guidance import action_has_target_contact, guided_shot_from_observation


class TrainedAgentPlayer:
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path or os.path.join("checkpoints", "snooker_ppo_latest.pt")
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

        if not os.path.exists(self.checkpoint_path):
            self.error = f"Using built-in AI: missing checkpoint {self.checkpoint_path}"
            self.ready = True
            return

        try:
            payload = torch.load(self.checkpoint_path, map_location="cpu")
            model_state = payload.get("model")
            if not isinstance(model_state, dict):
                self.error = "Using built-in AI: checkpoint does not contain a model."
                self.ready = True
                return
            if not self._state_is_finite(model_state):
                self.error = "Using built-in AI: checkpoint contains invalid model values."
                self.ready = True
                return

            first_layer = model_state.get("shared.0.weight")
            if first_layer is None:
                self.error = "Using built-in AI: could not read model input size from checkpoint."
                self.ready = True
                return

            self.state_dim = int(first_layer.shape[1])
            self.state_object_balls = self._ball_count_from_state_dim(self.state_dim)
            self.ball_count = self._payload_ball_count(payload, self.state_object_balls)

            cfg = PPOConfig()
            self.agent = PPO_Agent(self.state_dim, cfg)
            self.agent.net.load_state_dict(model_state)
            self.ready = True
            self.using_policy = True
        except Exception as exc:
            self.error = f"Using built-in AI: {exc}"
            self.ready = True

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
        return guided_shot_from_observation(obs, WIDTH, HEIGHT, holes)

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
            if angle_gap <= 0.55 or abs(power - fallback_power) <= 3.0:
                self.error = None
                return action

        if self.using_policy:
            self.error = "Using guided AI shot"
        return fallback

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
        }
