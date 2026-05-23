import pygame
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union


class Ball:
    """Ball class representing a pool ball in the environment."""
    def __init__(self, x, y, color, number=0, radius=12, is_striped=False):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.number = number
        self.vx = 0
        self.vy = 0
        self.friction = 0.98
        self.potted = False
        self.original_pos = (x, y)
        self.is_striped = is_striped
        self.shadow_offset = 3
    
    def move(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= self.friction
        self.vy *= self.friction
        if abs(self.vx) < 0.1: self.vx = 0
        if abs(self.vy) < 0.1: self.vy = 0
        
        return self.vx != 0 or self.vy != 0  # Return True if still moving
    
    def reset(self):
        self.x, self.y = self.original_pos
        self.vx, self.vy = 0, 0
        self.potted = False
    
    def strike(self, angle, power):
        self.vx = power * math.cos(angle)
        self.vy = power * math.sin(angle)
        return True  # Return True indicating successful strike
    
    def in_hole(self, holes, hole_radius):
        for hx, hy in holes:
            if math.hypot(self.x - hx, self.y - hy) < self.pocket_capture_radius(hole_radius):
                self.potted = True
                return True
        return False

    def pocket_capture_radius(self, hole_radius):
        return hole_radius + self.radius * 1.5
    
    def get_state(self):
        """Return a dictionary representing the ball's state."""
        return {
            'x': self.x,
            'y': self.y,
            'vx': self.vx,
            'vy': self.vy,
            'number': self.number,
            'potted': self.potted,
            'is_striped': self.is_striped
        }


# Constants
WIDTH, HEIGHT = 1200, 800
BALL_RADIUS = 12
HOLE_RADIUS = 20

# Colors for visualization
WHITE = (255, 255, 255)
RED = (200, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)


class PoolEnvironment:
    """Pool game environment for reinforcement learning."""
    
    def __init__(self, render_mode=None, ball_count: int = 15, random_balls: bool = False, layout: str = "rack"):
        """Initialize the pool environment.
        
        Args:
            render_mode: Optional rendering mode ('human', 'rgb_array', or None)
        """
        # Environment dimensions
        self.width = WIDTH
        self.height = HEIGHT
        self.ball_radius = BALL_RADIUS
        self.hole_radius = HOLE_RADIUS
        
        # Holes position
        self.holes = [
            (80, 80),           # Top left
            (WIDTH//2, 70),     # Top middle  
            (WIDTH-80, 80),     # Top right
            (80, HEIGHT-80),    # Bottom left
            (WIDTH//2, HEIGHT-70), # Bottom middle
            (WIDTH-80, HEIGHT-80)  # Bottom right
        ]
        
        # Game state variables
        self.cue_ball = None
        self.balls = []
        self.ball_count = self._clamp_ball_count(ball_count)
        self.shots_taken = 0
        self.random_balls = bool(random_balls)
        self.layout = self._normalize_layout(layout)
        self.max_shots = self._shot_limit()
        self.players = [
            {"name": "Player 1", "type": None, "score": 0, "color": (200, 30, 30)},
            {"name": "Player 2", "type": None, "score": 0, "color": (30, 150, 30)}
        ]
        self.current_player = 0
        self.turn_ended = False
        self.balls_moving = False
        self.foul = False
        self.ball_assignment_done = False
        self.first_ball_hit = None
        self.winner = None
        self.shots_taken = 0
        
        # Rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Reset to initialize the game
        self.reset()
    
    def _clamp_ball_count(self, count: int) -> int:
        return max(2, min(15, int(count)))

    def set_ball_count(self, count: int):
        self.ball_count = self._clamp_ball_count(count)
        self.max_shots = self._shot_limit()

    def _shot_limit(self):
        if self.layout == "beginner":
            return max(10, self.ball_count * 4)
        return max(30, self.ball_count * 12)

    def set_random_balls(self, enabled: bool):
        self.random_balls = bool(enabled)

    def set_layout(self, layout: str):
        self.layout = self._normalize_layout(layout)
        self.max_shots = self._shot_limit()

    def _normalize_layout(self, layout: str) -> str:
        layout = str(layout or "rack").lower()
        if layout in ("beginner", "easy"):
            return "beginner"
        return "rack"

    def _ball_colors(self):
        brown = (150, 75, 0)
        return [
            YELLOW,     # 1
            BLUE,       # 2
            RED,        # 3
            PURPLE,     # 4
            ORANGE,     # 5
            GREEN,      # 6
            brown,      # 7
            BLACK,      # 8
            YELLOW,     # 9  (striped)
            BLUE,       # 10 (striped)
            RED,        # 11 (striped)
            PURPLE,     # 12 (striped)
            ORANGE,     # 13 (striped)
            GREEN,      # 14 (striped)
            brown       # 15 (striped)
        ]

    def _random_position(self, existing):
        # Keep ball away from cushions/pockets and other balls
        min_x, max_x = 120, self.width - 120
        min_y, max_y = 120, self.height - 120
        for _ in range(100):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            ok = True
            for ball in existing:
                dist = math.hypot(x - ball.x, y - ball.y)
                if dist < (self.ball_radius + ball.radius + 6):
                    ok = False
                    break
            if ok:
                return x, y
        return None

    def _randomize_ball_positions(self, balls: list):
        placed = []
        for ball in balls:
            pos = self._random_position(placed)
            if pos is None:
                # Fallback: keep existing position
                placed.append(ball)
                continue
            ball.x, ball.y = pos
            ball.original_pos = (ball.x, ball.y)
            placed.append(ball)

    def _jitter_beginner_positions(self, balls: list):
        self.cue_ball.x = 280 + random.uniform(-35, 35)
        self.cue_ball.y = 213 + random.uniform(-35, 35)
        self.cue_ball.original_pos = (self.cue_ball.x, self.cue_ball.y)

        placed = [self.cue_ball]
        for ball in balls:
            original_x, original_y = ball.x, ball.y
            for _ in range(40):
                x = max(120, min(self.width - 120, original_x + random.uniform(-35, 35)))
                y = max(120, min(self.height - 120, original_y + random.uniform(-35, 35)))
                if all(math.hypot(x - other.x, y - other.y) >= self.ball_radius + other.radius + 8 for other in placed):
                    ball.x, ball.y = x, y
                    break
            ball.original_pos = (ball.x, ball.y)
            placed.append(ball)

    def setup_balls(self):
        """Set up the initial position of balls in triangle formation."""
        if self.layout == "beginner":
            return self._setup_beginner_balls()

        balls = []
        
        ball_colors = self._ball_colors()
        
        # Triangle rack formation
        rack_start_x = self.width * 3 // 4
        rack_start_y = self.height // 2
        ball_diameter = self.ball_radius * 2.2

        ball_positions = [
            (0, 0),                         # 1st row (apex)
            (1, -0.5), (1, 0.5),            # 2nd row
            (2, -1), (2, 0), (2, 1),        # 3rd row
            (3, -1.5), (3, -0.5), (3, 0.5), (3, 1.5),  # 4th row
            (4, -2), (4, -1), (4, 0), (4, 1), (4, 2)   # 5th row
        ]
        
        selected_positions = ball_positions[:self.ball_count]
        selected_numbers = self._select_training_ball_numbers()

        for i, (offset_x, offset_y) in enumerate(selected_positions):
            number = selected_numbers[i]
            x = rack_start_x + offset_x * ball_diameter
            y = rack_start_y + offset_y * ball_diameter
            is_striped = number > 8
            ball_number = number if number != 8 else 8
            balls.append(Ball(x, y, ball_colors[number-1], ball_number, self.ball_radius, is_striped))
        
        return balls

    def _setup_beginner_balls(self):
        """Set up balls in open, pottable positions for early training."""
        self.cue_ball.x = 280
        self.cue_ball.y = 213
        self.cue_ball.original_pos = (self.cue_ball.x, self.cue_ball.y)

        ball_colors = self._ball_colors()
        object_numbers = [n for n in range(1, 16) if n != 8]
        selected_numbers = object_numbers[:self.ball_count - 1] + [8]

        easy_positions = [
            (160, 133),                         # top-left pocket
            (self.width - 160, self.height - 133),  # bottom-right pocket
            (self.width - 160, 133),             # top-right pocket
            (160, self.height - 133),            # bottom-left pocket
            (self.width * 0.50, 130),            # top-middle pocket
            (self.width * 0.50, self.height - 130),  # bottom-middle pocket
            (self.width * 0.82, self.height * 0.50),
            (self.width * 0.18, self.height * 0.50),
            (self.width * 0.70, self.height * 0.34),
            (self.width * 0.30, self.height * 0.66),
            (self.width * 0.60, self.height * 0.24),
            (self.width * 0.40, self.height * 0.76),
            (self.width * 0.75, self.height * 0.76),
            (self.width * 0.25, self.height * 0.24),
            (self.width * 0.50, self.height * 0.50),
        ]

        balls = []
        for number, (x, y) in zip(selected_numbers, easy_positions):
            is_striped = number > 8
            balls.append(Ball(x, y, ball_colors[number - 1], number, self.ball_radius, is_striped))

        return balls

    def _select_training_ball_numbers(self):
        """Choose a reduced 8-ball set that always contains the 8 ball."""
        object_numbers = [n for n in range(1, 16) if n != 8]
        random.shuffle(object_numbers)

        if self.ball_count >= 5:
            selected = object_numbers[:self.ball_count - 1]
            selected.insert(4, 8)
            return selected

        selected = object_numbers[:self.ball_count - 1]
        selected.append(8)
        return selected
    
    def reset(self):
        """Reset the environment to initial state."""
        # Create cue ball
        self.cue_ball = Ball(self.width//4, self.height//2, WHITE, 0)
        
        # Create all other balls
        self.balls = self.setup_balls()

        if self.random_balls and self.layout == "beginner":
            self._jitter_beginner_positions(self.balls)
        elif self.random_balls:
            self._randomize_ball_positions(self.balls)
            pos = self._random_position(self.balls)
            if pos:
                self.cue_ball.x, self.cue_ball.y = pos
                self.cue_ball.original_pos = (self.cue_ball.x, self.cue_ball.y)
        
        # Reset game state
        self.players = [
            {"name": "Player 1", "type": None, "score": 0, "color": (200, 30, 30)},
            {"name": "Player 2", "type": None, "score": 0, "color": (30, 150, 30)}
        ]
        self.current_player = 0
        self.turn_ended = False
        self.balls_moving = False
        self.foul = False
        self.ball_assignment_done = False
        self.first_ball_hit = None
        self.winner = None
        self.shots_taken = 0
        
        # Return initial observation
        return self._get_observation()
    
    def _get_observation(self):
        """Return the current state observation."""
        ball_states = [self.cue_ball.get_state()]
        ball_states.extend([ball.get_state() for ball in self.balls])
        
        # Include game state information
        observation = {
            'balls': ball_states,
            'current_player': self.current_player,
            'player_types': [self.players[0]['type'], self.players[1]['type']],
            'ball_assignment_done': self.ball_assignment_done,
            'foul': self.foul,
            'turn_ended': self.turn_ended,
            'winner': self.winner
        }
        
        return observation
    
    def should_hit_8_ball(self):
        """Check if the current player should be hitting the 8-ball."""
        player_type = self.players[self.current_player]["type"]
        if player_type is None:
            return False
            
        # Check if all of the player's balls are potted
        player_balls_left = False
        for ball in self.balls:
            if not ball.potted and ball.number != 8:
                if (ball.is_striped and player_type == "striped") or (not ball.is_striped and player_type == "solid"):
                    player_balls_left = True
                    break
                    
        return not player_balls_left
    
    def handle_ball_collision(self, ball1, ball2):
        """Handle collision physics between two balls."""
        dx = ball2.x - ball1.x
        dy = ball2.y - ball1.y
        distance = math.hypot(dx, dy)
        min_dist = ball1.radius + ball2.radius
        
        if distance < min_dist:
            if distance <= 1e-6:
                rel_x = ball2.vx - ball1.vx
                rel_y = ball2.vy - ball1.vy
                angle = math.atan2(rel_y, rel_x) if abs(rel_x) > 1e-6 or abs(rel_y) > 1e-6 else 0.0
                nx = math.cos(angle)
                ny = math.sin(angle)
            else:
                nx = dx / distance
                ny = dy / distance

            overlap = min_dist - max(distance, 1e-6) + 0.01
            ball1.x -= overlap * nx / 2
            ball1.y -= overlap * ny / 2
            ball2.x += overlap * nx / 2
            ball2.y += overlap * ny / 2

            relative_speed = (ball2.vx - ball1.vx) * nx + (ball2.vy - ball1.vy) * ny
            if relative_speed >= 0:
                return False

            angle = math.atan2(ny, nx)
            v1 = math.hypot(ball1.vx, ball1.vy)
            v2 = math.hypot(ball2.vx, ball2.vy)
            dir1 = math.atan2(ball1.vy, ball1.vx) if v1 > 0 else 0
            dir2 = math.atan2(ball2.vy, ball2.vx) if v2 > 0 else 0

            new_x_vel1 = v2 * math.cos(dir2 - angle) * math.cos(angle)
            new_y_vel1 = v2 * math.cos(dir2 - angle) * math.sin(angle)
            new_x_vel2 = v1 * math.cos(dir1 - angle) * math.cos(angle)
            new_y_vel2 = v1 * math.cos(dir1 - angle) * math.sin(angle)

            ball1.vx = new_x_vel1 * 0.95
            ball1.vy = new_y_vel1 * 0.95
            ball2.vx = new_x_vel2 * 0.95
            ball2.vy = new_y_vel2 * 0.95

            ball1.vx += v1 * math.sin(dir1 - angle) * math.cos(angle + math.pi / 2) * 0.95
            ball1.vy += v1 * math.sin(dir1 - angle) * math.sin(angle + math.pi / 2) * 0.95
            ball2.vx += v2 * math.sin(dir2 - angle) * math.cos(angle + math.pi / 2) * 0.95
            ball2.vy += v2 * math.sin(dir2 - angle) * math.sin(angle + math.pi / 2) * 0.95

            return True  # Collision occurred
        return False  # No collision

    def handle_collisions(self):
        """Handle all ball-to-ball collisions in the environment."""
        collision_occurred = False
        
        # Ball-to-Ball collisions
        for i in range(len(self.balls)):
            if self.balls[i].potted:
                continue
            
            # Check cue ball collision
            if not self.cue_ball.potted:
                if self.handle_ball_collision(self.cue_ball, self.balls[i]):
                    collision_occurred = True
                    if not self.first_ball_hit:
                        self.first_ball_hit = self.balls[i]
            
            # Check other ball collisions
            for j in range(i + 1, len(self.balls)):
                if not self.balls[j].potted:
                    if self.handle_ball_collision(self.balls[i], self.balls[j]):
                        collision_occurred = True
        
        return collision_occurred
    
    def check_cushion_collisions(self, ball):
        """Handle ball collisions with table cushions."""
        collision = False
        cushion_dampening = 0.8  # Energy loss on cushion hit

        for hx, hy in self.holes:
            if math.hypot(ball.x - hx, ball.y - hy) < ball.pocket_capture_radius(self.hole_radius):
                ball.vx = 0
                ball.vy = 0
                return False
        
        if ball.x - ball.radius < 80:
            ball.x = 80 + ball.radius
            ball.vx = -ball.vx * cushion_dampening
            collision = True
        elif ball.x + ball.radius > self.width - 80:
            ball.x = self.width - 80 - ball.radius
            ball.vx = -ball.vx * cushion_dampening
            collision = True
            
        if ball.y - ball.radius < 80:
            ball.y = 80 + ball.radius
            ball.vy = -ball.vy * cushion_dampening
            collision = True
        elif ball.y + ball.radius > self.height - 80:
            ball.y = self.height - 80 - ball.radius
            ball.vy = -ball.vy * cushion_dampening
            collision = True

        if abs(ball.vx) < 0.12:
            ball.vx = 0
        if abs(ball.vy) < 0.12:
            ball.vy = 0
            
        return collision
    
    def valid_shot(self):
        """Check if the shot is valid according to 8-ball rules."""
        # If no ball was hit
        if not self.first_ball_hit:
            return False
        
        # If player types aren't assigned yet, any hit is valid
        if not self.ball_assignment_done:
            return True
        
        # Get player's ball type
        player_type = self.players[self.current_player]["type"]
        
        # If player should hit 8-ball
        if self.should_hit_8_ball():
            return self.first_ball_hit.number == 8
        
        # Otherwise player should hit their type
        first_hit_type = "striped" if self.first_ball_hit.is_striped else "solid"
        return first_hit_type == player_type
    
    def process_potted_ball(self, ball):
        """Process a ball that has been potted."""
        # If this is the first assignment
        if not self.ball_assignment_done and ball.number != 8:
            # Assign ball types to players
            current_player_type = "striped" if ball.is_striped else "solid"
            other_player_type = "solid" if ball.is_striped else "striped"
            
            self.players[self.current_player]["type"] = current_player_type
            self.players[1 - self.current_player]["type"] = other_player_type
            self.ball_assignment_done = True
            
            # Player keeps turn after their first assignment
            self.turn_ended = False
            
        # Handle 8-ball special case
        elif ball.number == 8:
            # If all player's balls are potted, they win
            if self.should_hit_8_ball():
                self.winner = self.current_player
            else:
                # Potting 8-ball too early loses the game
                self.winner = 1 - self.current_player
            self.turn_ended = True
            
        # Handle standard potting
        else:
            # If ball types are assigned
            if self.ball_assignment_done:
                player_type = self.players[self.current_player]["type"]
                ball_type = "striped" if ball.is_striped else "solid"
                
                # If player potted their own ball type
                if player_type == ball_type:
                    self.turn_ended = False  # Player continues their turn
                else:
                    self.turn_ended = True  # Player potted opponent's ball - end turn
            else:
                # If ball types not yet assigned, count as a foul
                self.turn_ended = True
    
    def end_turn(self):
        """End the current player's turn and switch to the next player."""
        # Reset cue ball if it was potted (ball in hand)
        if self.cue_ball.potted:
            self.cue_ball.x = self.width // 4
            self.cue_ball.y = self.height // 2
            self.cue_ball.vx = 0
            self.cue_ball.vy = 0
            self.cue_ball.potted = False
        
        # Switch players if turn ended
        if self.turn_ended or self.foul:
            self.current_player = 1 - self.current_player
            self.foul = False
            self.turn_ended = False
            self.first_ball_hit = None

    def _is_target_ball_for_player(self, ball, player_index):
        """Return True if `ball` is a legal target for `player_index` right now."""
        if ball.number == 8:
            player_type = self.players[player_index]["type"]
            if player_type is None:
                return False
            player_balls_left = False
            for b in self.balls:
                if b.potted or b.number == 8:
                    continue
                b_type = "striped" if b.is_striped else "solid"
                if b_type == player_type:
                    player_balls_left = True
                    break
            return not player_balls_left

        if not self.ball_assignment_done:
            return True

        player_type = self.players[player_index]["type"]
        if player_type is None:
            return False
        ball_type = "striped" if ball.is_striped else "solid"
        return ball_type == player_type

    def _best_target_distance_to_hole(self, player_index):
        """Closest distance from any current target ball to any pocket."""
        best = None
        for ball in self.balls:
            if ball.potted or ball.number == 8:
                continue
            if not self._is_target_ball_for_player(ball, player_index):
                continue
            d = min(math.hypot(ball.x - hx, ball.y - hy) for hx, hy in self.holes)
            if best is None or d < best:
                best = d
        return best
    
    def step(self, action):
        """Take a step in the environment with the given action.
        
        Args:
            action: Dictionary with 'angle' and 'power' for the cue strike
            
        Returns:
            observation: The new state of the environment
            reward: Reward for the current step
            done: Whether the episode is finished
            info: Additional information
        """
        # Extract action parameters
        acting_player = self.current_player
        assignment_done_before_shot = self.ball_assignment_done
        angle = action.get('angle', 0)
        power = action.get('power', 0)
        
        reward = 0
        done = False
        info = {}
        shot_potted_balls = []
        self.shots_taken += 1
        best_target_dist_before = self._best_target_distance_to_hole(acting_player)
        reward -= 0.05  # Small shot cost encourages finishing in fewer turns
        
        # Apply the action: strike the cue ball
        if not self.balls_moving and not self.cue_ball.potted:
            self.cue_ball.strike(angle, power)
            self.balls_moving = True
            self.first_ball_hit = None
        
        # Simulate physics until balls stop moving
        simulation_steps = 0
        max_simulation_steps = 2000
        while self.balls_moving:
            simulation_steps += 1
            # Move all balls
            still_moving = False
            
            # Move cue ball
            if not self.cue_ball.potted:
                if self.cue_ball.move():
                    still_moving = True
                self.check_cushion_collisions(self.cue_ball)
            
            # Move other balls
            for ball in self.balls:
                if not ball.potted:
                    if ball.move():
                        still_moving = True
                    self.check_cushion_collisions(ball)
            
            # Handle collisions
            if self.handle_collisions():
                still_moving = True
            
            # Check for potted balls
            potted_balls = []
            for ball in self.balls:
                if not ball.potted and ball.in_hole(self.holes, self.hole_radius):
                    potted_balls.append(ball)
                    shot_potted_balls.append(ball)
                    reward += 1  # Reward for potting a ball
            
            # Process all potted balls
            for ball in potted_balls:
                self.process_potted_ball(ball)
            
            # Check if cue ball is potted
            if not self.cue_ball.potted and self.cue_ball.in_hole(self.holes, self.hole_radius):
                self.foul = True
                self.cue_ball.potted = True
                self.turn_ended = True
                reward -= 5  # Penalty for potting the cue ball
            
            # Update the balls_moving flag
            self.balls_moving = still_moving
            
            # If no balls are moving, exit the simulation loop
            if not still_moving:
                break

            if simulation_steps >= max_simulation_steps:
                self.balls_moving = False
                self.cue_ball.vx = 0
                self.cue_ball.vy = 0
                for ball in self.balls:
                    ball.vx = 0
                    ball.vy = 0
                break
        
        # After simulation, check if a valid shot was made
        if self.first_ball_hit is None and shot_potted_balls:
            self.first_ball_hit = shot_potted_balls[0]

        shot_valid = self.valid_shot()
        if not assignment_done_before_shot and self.first_ball_hit is not None:
            shot_valid = True

        if not self.foul and not shot_valid:
            self.foul = True
            reward -= 2  # Penalty for invalid shot

        # Reward/Penalty for first contact quality
        if self.first_ball_hit is None:
            reward -= 0.5
        elif self._is_target_ball_for_player(self.first_ball_hit, acting_player):
            reward += 0.35
        else:
            reward -= 0.75

        # Reward shaping for own/opponent pots when assignments are known
        if self.ball_assignment_done:
            for ball in shot_potted_balls:
                if ball.number == 8:
                    continue
                if self._is_target_ball_for_player(ball, acting_player):
                    reward += 0.75
                else:
                    reward -= 1.0

        if self.layout == "beginner" or (self.layout == "rack" and self.ball_count <= 6):
            object_ball_potted = any(ball.number != 8 for ball in shot_potted_balls)
            potted_object_balls = sum(
                1 for ball in self.balls if ball.number != 8 and ball.potted
            )
            if self.ball_count <= 4:
                required_object_pots = 1
            elif self.ball_count <= 6:
                required_object_pots = 2
            else:
                required_object_pots = min(3, self.ball_count - 1)
            if (
                (self.ball_count <= 4 and object_ball_potted)
                or (self.ball_count > 4 and potted_object_balls >= required_object_pots)
            ):
                self.winner = acting_player
                done = True
                info["beginner_drill_complete"] = True

        # Reward making target balls closer to pockets; penalty for moving away
        best_target_dist_after = self._best_target_distance_to_hole(acting_player)
        if best_target_dist_before is not None and best_target_dist_after is not None:
            progress = best_target_dist_before - best_target_dist_after
            reward += float(np.clip(progress / 120.0, -0.6, 0.6))
        
        if self.turn_ended or self.foul:
            self.end_turn()
        
        # If all balls are potted, end the game
        if all(b.potted for b in self.balls):
            self.winner = acting_player
            done = True

        if self.winner is None and self.shots_taken >= self.max_shots:
            self.winner = 1
            done = True
            info["timeout"] = True
            reward -= 8

        # Check for game over
        if self.winner is not None:
            done = True
            if self.winner == acting_player:
                reward += 30  # Big reward for winning
            else:
                reward -= 30  # Big penalty for losing

        info["first_ball_hit"] = self.first_ball_hit.number if self.first_ball_hit else None
        info["potted_count"] = len(shot_potted_balls)
        
        # Return the step results
        return self._get_observation(), reward, done, info
    
    def get_legal_actions(self):
        """Return a set of legal actions from the current state.
        
        In pool, legal actions are all possible angles and power levels for the cue.
        This is a continuous action space, so we return the bounds.
        """
        if self.balls_moving or self.cue_ball.potted:
            return []  # No actions allowed while balls are moving or cue ball is potted
        
        # Return the bounds for angle (0 to 2π) and power (0 to max_power)
        return {
            'angle': (0, 2 * math.pi),
            'power': (0, 25)  # Max power of 25 from original game
        }
    
    def count_remaining_balls(self, player_index):
        """Count how many balls a player still needs to pot."""
        count = 0
        player_type = self.players[player_index]["type"]
        
        if player_type is None:
            return sum(1 for ball in self.balls if not ball.potted and ball.number != 8)
        
        for ball in self.balls:
            if not ball.potted and ball.number != 8:
                ball_type = "striped" if ball.is_striped else "solid"
                if ball_type == player_type:
                    count += 1
        
        return count
    
    def render(self, mode=None):
        """Render the current state of the environment.
        
        Args:
            mode: 'human' to display to a window, 'rgb_array' to return an array
        """
        if mode is None:
            mode = self.render_mode
            
        if mode is None:
            return
            
        # Initialize pygame if not done yet
        if self.screen is None and mode == 'human':
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pool Environment")
            self.clock = pygame.time.Clock()
            
        # Create a surface if returning an array
        if mode == 'rgb_array' and self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.Surface((self.width, self.height))
            
        # Fill the background
        self.screen.fill((50, 30, 10))  # Dark wood color
        
        # Draw the table
        pygame.draw.rect(self.screen, (0, 100, 0), (60, 60, self.width-120, self.height-120), 0)
        pygame.draw.rect(self.screen, (0, 128, 0), (80, 80, self.width-160, self.height-160), 0)
        
        # Draw pockets/holes
        for hx, hy in self.holes:
            pygame.draw.circle(self.screen, (10, 10, 10), (hx, hy), self.hole_radius)
            pygame.draw.circle(self.screen, (50, 30, 10), (hx, hy), self.hole_radius+5, 5)
        
        # Draw balls
        for ball in self.balls:
            if not ball.potted:
                pygame.draw.circle(self.screen, ball.color, (int(ball.x), int(ball.y)), ball.radius)
                # Add number to the ball
                if ball.number > 0:
                    text_color = WHITE if ball.color == BLACK or ball.color == BLUE else BLACK
                    number_font = pygame.font.SysFont('Arial', 10, bold=True)
                    number_text = number_font.render(str(ball.number), True, text_color)
                    text_rect = number_text.get_rect(center=(int(ball.x), int(ball.y)))
                    self.screen.blit(number_text, text_rect)
        
        # Draw cue ball
        if not self.cue_ball.potted:
            pygame.draw.circle(self.screen, self.cue_ball.color, (int(self.cue_ball.x), int(self.cue_ball.y)), self.cue_ball.radius)
        
        # Display current player and game state
        font = pygame.font.SysFont('Arial', 24)
        current_player_text = font.render(f"Current Player: {self.players[self.current_player]['name']}", True, WHITE)
        self.screen.blit(current_player_text, (10, 10))
        
        if self.ball_assignment_done:
            player_type = font.render(f"Type: {self.players[self.current_player]['type']}", True, WHITE)
            self.screen.blit(player_type, (10, 40))
            
        if self.foul:
            foul_text = font.render("FOUL!", True, RED)
            self.screen.blit(foul_text, (self.width//2 - foul_text.get_width()//2, 10))
            
        if self.winner is not None:
            winner_text = font.render(f"{self.players[self.winner]['name']} WINS!", True, WHITE)
            self.screen.blit(winner_text, (self.width//2 - winner_text.get_width()//2, self.height//2))
        
        # Update display
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(60)
            
        # Return the array if needed
        if mode == 'rgb_array':
            return pygame.surfarray.array3d(self.screen)
    
    def close(self):
        """Close the environment."""
        if self.screen is not None:
            self.screen = None
        self.clock = None


# Example usage
if __name__ == "__main__":
    env = PoolEnvironment(render_mode='human')
    observation = env.reset()
    
    # Main loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # Simple manual control for testing
            if event.type == pygame.MOUSEBUTTONDOWN and not env.balls_moving:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                dx = mouse_x - env.cue_ball.x
                dy = mouse_y - env.cue_ball.y
                angle = math.atan2(dy, dx)
                power = min(math.hypot(dx, dy) / 10, 25)
                
                # Take action
                observation, reward, done, info = env.step({'angle': angle, 'power': power})
                print(f"Reward: {reward}, Done: {done}")
                
                if done:
                    print("Game over!")
                    observation = env.reset()
        
        # Update environment simulation
        if env.balls_moving:
            # Simulate the physics (this would be in step for an RL agent)
            # Here we just update the rendering for manual testing
            env.render()
        
        # Keep rendering
        env.render()
    
    env.close()
