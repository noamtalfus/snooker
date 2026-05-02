import os

from game_config import HEIGHT, WIDTH


class TrainedAgentPlayer:
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path or os.path.join("checkpoints", "snooker_ppo_latest.pt")
        self.ready = False
        self.error = None
        self.ball_count = 15
        self.state_dim = None

        try:
            import torch
            from ppo import PPOConfig, PPO_Agent, obs_to_compact_state
        except Exception as exc:
            self.error = f"AI load failed: {exc}"
            return

        self.torch = torch
        self.obs_to_compact_state = obs_to_compact_state

        if not os.path.exists(self.checkpoint_path):
            self.error = f"Missing checkpoint: {self.checkpoint_path}"
            return

        try:
            payload = torch.load(self.checkpoint_path, map_location="cpu")
            model_state = payload.get("model")
            if not isinstance(model_state, dict):
                self.error = "Checkpoint does not contain a model."
                return
            if not self._state_is_finite(model_state):
                self.error = "Checkpoint contains invalid model values."
                return

            first_layer = model_state.get("shared.0.weight")
            if first_layer is None:
                self.error = "Could not read model input size from checkpoint."
                return

            self.state_dim = int(first_layer.shape[1])
            self.ball_count = self._ball_count_from_state_dim(self.state_dim)

            cfg = PPOConfig()
            self.agent = PPO_Agent(self.state_dim, cfg)
            self.agent.net.load_state_dict(model_state)
            self.ready = True
        except Exception as exc:
            self.error = f"AI load failed: {exc}"

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

    def choose_action(self, game):
        obs = self._game_to_observation(game)
        state_vec = self.obs_to_compact_state(obs, WIDTH, HEIGHT)
        if state_vec.shape[0] != self.state_dim:
            raise ValueError(
                f"AI expects {self.ball_count} balls, but current state has a different size."
            )
        action, _, _, _ = self.agent.select_action(state_vec, deterministic=True)
        return action

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
