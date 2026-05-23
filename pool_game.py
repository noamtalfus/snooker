import math
import random

import pygame

from game_config import *
from game_objects import Ball, Cue
from training_controller import TrainingController
from trained_agent_player import TrainedAgentPlayer

class PoolGame:
    def __init__(self):
        self.trained_agent = None
        self.vs_trained_agent = False
        self.ai_player_index = 1
        self.ai_shot_delay = 0
        self.ai_status_text = ""
        self.holes = holes
        self.active_ball_count = 15
        self.graphics_style = "classic"
        self.is_fullscreen = IS_FULLSCREEN
        self.display_screen = screen
        self.reset_game()
        self.state = MENU
        self.menu_options = ["New Game", "Play vs Trained AI", "Training Mode", "Instructions", "Graphics Style", "Fullscreen", "Quit"]
        self.selected_option = 0
        self.training = None
        self.training_selected_option = 0
        self.training_ball_count = 15
        self.training_layout = "rack"
        self.training_random_balls = False
        self.training_opponent = "random"  # random | self | none
        self.training_workout_plan = False
        self.training_weird_graphics = False
        self.training_dropdown_open = None
        self.training_control_rects = {}
        self.menu_item_rects = []
        self.instruction_back_rect = None
       
    def reset_game(self, ball_count=15, vs_trained_agent=False):
        self.active_ball_count = max(2, min(15, int(ball_count)))
        self.vs_trained_agent = bool(vs_trained_agent)
        self.ai_shot_delay = 45 if self.vs_trained_agent else 0
        if not self.vs_trained_agent:
            self.ai_status_text = ""
        self.cue_ball = Ball(WIDTH//4, HEIGHT//2, WHITE, 0)
        self.balls = self.setup_balls()
        self.cue = Cue()
        self.players = [
            {"name": "Player 1", "type": None, "score": 0, "color": PLAYER1_COLOR},
            {"name": "Trained AI" if self.vs_trained_agent else "Player 2", "type": None, "score": 0, "color": PLAYER2_COLOR}
        ]
        self.current_player = 0
        self.turn_ended = False
        self.shot_potted_ball = False
        self.balls_moving = False
        self.last_potted = None
        self.foul = False
        self.ball_assignment_done = False  # Track if balls have been assigned to players
        self.target_ball = None  # 8-ball in the final stage
        self.first_ball_hit = None
        self.winner = None
        self.shot_motion_frames = 0
       
    def setup_balls(self):
        balls = []
       
        
        ball_colors = [
            YELLOW,     # 1
            BLUE,       # 2
            RED,        # 3
            PURPLE,     # 4
            ORANGE,     # 5
            GREEN,      # 6
            BROWN,      # 7
            BLACK,      # 8  (8 ball)  
            YELLOW,     # 9  (striped)
            BLUE,       # 10 (striped)
            RED,        # 11 (striped)
            PURPLE,     # 12 (striped)
            ORANGE,     # 13 (striped)
            GREEN,      # 14 (striped)
            BROWN       # 15 (striped)
        ]
       
        rack_start_x = WIDTH * 3 // 4
        rack_start_y = HEIGHT // 2
        ball_diameter = BALL_RADIUS * 2.2


        ball_positions = [
            (0, 0),                         # 1st row (apex)
            (1, -0.5), (1, 0.5),            # 2nd row
            (2, -1), (2, 0), (2, 1),        # 3rd row
            (3, -1.5), (3, -0.5), (3, 0.5), (3, 1.5),  # 4th row
            (4, -2), (4, -1), (4, 0), (4, 1), (4, 2)   # 5th row
        ]
       
        
        object_numbers = [n for n in range(1, 16) if n != 8]
        random.shuffle(object_numbers)
        if self.active_ball_count >= 5:
            selected_numbers = object_numbers[:self.active_ball_count - 1]
            selected_numbers.insert(4, 8)
        else:
            selected_numbers = object_numbers[:self.active_ball_count - 1]
            selected_numbers.append(8)
       
        selected_positions = ball_positions[:self.active_ball_count]

        for i, (offset_x, offset_y) in enumerate(selected_positions):
            number = selected_numbers[i]
            x = rack_start_x + offset_x * ball_diameter
            y = rack_start_y + offset_y * ball_diameter
            is_striped = number > 8
            ball_number = number if number != 8 else 8
            balls.append(Ball(x, y, ball_colors[number-1], ball_number, BALL_RADIUS, is_striped))
       
        return balls
   
    def handle_menu_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, rect in enumerate(self.menu_item_rects):
                if rect.collidepoint(event.pos):
                    self.selected_option = index
                    if index == 0:
                        self.reset_game(15, False)
                        self.state = GAME
                    elif index == 1:
                        self.start_human_vs_agent()
                    elif index == 2:
                        self.state = TRAINING_MENU
                    elif index == 3:
                        self.state = INSTRUCTIONS
                    elif index == 4:
                        self.toggle_graphics_style()
                    elif index == 5:
                        self.toggle_fullscreen()
                    elif index == 6:
                        return False
                    break

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_option = (self.selected_option - 1) % len(self.menu_options)
            elif event.key == pygame.K_DOWN:
                self.selected_option = (self.selected_option + 1) % len(self.menu_options)
            elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                if self.selected_option == 4:
                    self.toggle_graphics_style()
                elif self.selected_option == 5:
                    self.toggle_fullscreen()
            elif event.key == pygame.K_F11:
                self.toggle_fullscreen()
            elif event.key == pygame.K_RETURN:
                if self.selected_option == 0:
                    self.reset_game(15, False)
                    self.state = GAME
                elif self.selected_option == 1:
                    self.start_human_vs_agent()
                elif self.selected_option == 2:
                    self.state = TRAINING_MENU
                elif self.selected_option == 3:
                    self.state = INSTRUCTIONS
                elif self.selected_option == 4:
                    self.toggle_graphics_style()
                elif self.selected_option == 5:
                    self.toggle_fullscreen()
                elif self.selected_option == 6:
                    return False  # Quit
        return True
   
    def handle_game_events(self, event):
        if self.is_ai_turn():
            return True

        if not self.balls_moving and not self.turn_ended:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                dx = self.cue_ball.x - mouse_x
                dy = self.cue_ball.y - mouse_y
                distance = math.hypot(dx, dy)

                self.cue.angle = math.atan2(dy, dx)
                self.cue.power = min(distance / BALL_RADIUS, self.cue.max_power)
                self.cue.pulling_back = True

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.cue.pulling_back:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                dx = mouse_x - self.cue_ball.x
                dy = mouse_y - self.cue_ball.y
                angle = math.atan2(dy, dx)
                self.cue_ball.strike(angle, self.cue.power)
                self.cue.pulling_back = False
                self.balls_moving = True
                self.shot_potted_ball = False
                self.first_ball_hit = None
                self.shot_motion_frames = 0




            elif event.type == pygame.MOUSEMOTION and self.cue.pulling_back:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                dx = mouse_x - self.cue_ball.x
                dy = mouse_y - self.cue_ball.y
                distance = math.hypot(dx, dy)
                self.cue.power = min(distance / 10, self.cue.max_power)
                self.cue.angle = math.atan2(dy, dx)

        return True


               
    def handle_paused_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.state = GAME
            elif event.key == pygame.K_m:
                self.state = MENU
        return True
   
    def handle_instructions_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.instruction_back_rect and self.instruction_back_rect.collidepoint(event.pos):
                self.state = MENU
                return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                self.state = MENU
        return True
   
    def handle_game_over_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.state = MENU
        return True

    def close_training(self):
        if self.training:
            self.training.close()
            self.training = None

    def start_human_vs_agent(self):
        if self.trained_agent is None or not self.trained_agent.ready:
            self.trained_agent = TrainedAgentPlayer()

        if not self.trained_agent.ready:
            self.ai_status_text = self.trained_agent.error or "Trained AI is not ready."
            return

        label = "trained AI" if self.trained_agent.using_policy else "stable shot planner"
        self.ai_status_text = f"Playing against {label} (15 balls)"
        self.reset_game(15, True)
        self.state = GAME

    def start_training(self, ball_count, random_balls, opponent_type, layout, workout_plan):
        self.close_training()
        self.training = TrainingController(
            ball_count=ball_count,
            random_balls=random_balls,
            opponent_type=opponent_type,
            resume_checkpoint=workout_plan,
            layout=layout,
            workout_plan=workout_plan
        )
        self.state = TRAINING

    def handle_training_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.close_training()
                self.state = MENU
            elif event.key == pygame.K_SPACE and self.training:
                self.training.toggle_pause()
        return True

    def handle_training_menu_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            self.training_dropdown_open = None if self.training_dropdown_open == "close" else self.training_dropdown_open

            back_rect = self.training_control_rects.get("back")
            if back_rect and back_rect.collidepoint(mouse_pos):
                self.state = MENU
                return True

            slider_rect = self.training_control_rects.get("ball_slider")
            if slider_rect and slider_rect.collidepoint(mouse_pos):
                t = (mouse_pos[0] - slider_rect.left) / max(1, slider_rect.width)
                self.training_ball_count = max(2, min(15, int(round(2 + t * 13))))
                self.training_selected_option = 0
                return True

            for value, rect in self.training_control_rects.get("layout_options", []):
                if rect.collidepoint(mouse_pos):
                    self.training_layout = value
                    self.training_selected_option = 1
                    return True

            for value, rect in self.training_control_rects.get("opponent_options", []):
                if rect.collidepoint(mouse_pos):
                    self.training_opponent = value
                    self.training_selected_option = 3
                    return True

            for key in ("random", "workout", "weird"):
                rect = self.training_control_rects.get(key)
                if rect and rect.collidepoint(mouse_pos):
                    if key == "random":
                        self.training_random_balls = not self.training_random_balls
                        self.training_selected_option = 2
                    elif key == "workout":
                        self.training_workout_plan = not self.training_workout_plan
                        self.training_selected_option = 4
                    else:
                        self.training_weird_graphics = not self.training_weird_graphics
                        self.training_selected_option = 5
                    return True

            for key in ("layout", "opponent"):
                rect = self.training_control_rects.get(key)
                if rect and rect.collidepoint(mouse_pos):
                    self.training_dropdown_open = None if self.training_dropdown_open == key else key
                    return True

            if self.training_dropdown_open:
                options = self.training_control_rects.get(f"{self.training_dropdown_open}_options", [])
                for value, rect in options:
                    if rect.collidepoint(mouse_pos):
                        if self.training_dropdown_open == "layout":
                            self.training_layout = value
                        elif self.training_dropdown_open == "opponent":
                            self.training_opponent = value
                        self.training_dropdown_open = None
                        return True

            start_rect = self.training_control_rects.get("start")
            if start_rect and start_rect.collidepoint(mouse_pos):
                self.start_training(
                    self.training_ball_count,
                    self.training_random_balls,
                    self.training_opponent,
                    self.training_layout,
                    self.training_workout_plan
                )
                return True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.training_selected_option = (self.training_selected_option - 1) % 7
            elif event.key == pygame.K_DOWN:
                self.training_selected_option = (self.training_selected_option + 1) % 7
            elif event.key == pygame.K_ESCAPE:
                self.state = MENU
            elif event.key == pygame.K_LEFT:
                if self.training_selected_option == 0:
                    self.training_ball_count = max(2, self.training_ball_count - 1)
                elif self.training_selected_option == 1:
                    self.training_layout = self._cycle_training_layout()
                elif self.training_selected_option == 2:
                    self.training_random_balls = not self.training_random_balls
                elif self.training_selected_option == 3:
                    self.training_opponent = self._cycle_opponent(-1)
                elif self.training_selected_option == 4:
                    self.training_workout_plan = not self.training_workout_plan
                elif self.training_selected_option == 5:
                    self.training_weird_graphics = not self.training_weird_graphics
            elif event.key == pygame.K_RIGHT:
                if self.training_selected_option == 0:
                    self.training_ball_count = min(15, self.training_ball_count + 1)
                elif self.training_selected_option == 1:
                    self.training_layout = self._cycle_training_layout()
                elif self.training_selected_option == 2:
                    self.training_random_balls = not self.training_random_balls
                elif self.training_selected_option == 3:
                    self.training_opponent = self._cycle_opponent(1)
                elif self.training_selected_option == 4:
                    self.training_workout_plan = not self.training_workout_plan
                elif self.training_selected_option == 5:
                    self.training_weird_graphics = not self.training_weird_graphics
            elif event.key == pygame.K_RETURN:
                if self.training_selected_option == 5:
                    self.training_weird_graphics = not self.training_weird_graphics
                elif self.training_selected_option == 6:
                    self.start_training(
                        self.training_ball_count,
                        self.training_random_balls,
                        self.training_opponent,
                        self.training_layout,
                        self.training_workout_plan
                    )
        return True

    def is_ai_turn(self):
        return (
            self.vs_trained_agent
            and self.current_player == self.ai_player_index
            and self.state == GAME
            and self.winner is None
        )

    def take_ai_shot(self):
        if not self.trained_agent or not self.trained_agent.ready:
            return
        if self.ai_shot_delay > 0:
            self.ai_shot_delay -= 1
            return

        try:
            action = self.trained_agent.choose_action(self)
        except Exception as exc:
            self.ai_status_text = f"AI shot failed: {exc}"
            self.ai_shot_delay = 60
            return

        angle = float(action.get("angle", 0.0))
        power = max(0.0, min(float(action.get("power", 0.0)), self.cue.max_power))
        self.cue.angle = angle
        self.cue.power = power
        self.cue_ball.strike(angle, power)
        self.cue.pulling_back = False
        self.balls_moving = True
        self.shot_potted_ball = False
        self.first_ball_hit = None
        self.shot_motion_frames = 0
        self.ai_shot_delay = 45
   
    def update_game(self):
        if self.is_ai_turn() and not self.balls_moving and not self.turn_ended:
            self.take_ai_shot()

        if self.balls_moving:
            self.shot_motion_frames += 1
            
            collided_balls = self.check_first_collision()
            if collided_balls and not self.first_ball_hit:
                self.first_ball_hit = collided_balls
           
            
            for ball in [self.cue_ball] + self.balls:
                if not ball.potted:
                    ball.move()
           
            self.handle_collisions()
            self.settle_slow_balls()
            all_stopped = not self.any_ball_moving()
           
            
            potted_balls = []
            for ball in self.balls:
                if not ball.potted and ball.in_hole():
                    potted_balls.append(ball)
           
            # Process all potted balls
            for ball in potted_balls:
                self.shot_potted_ball = True
                self.process_potted_ball(ball)

            if self.first_ball_hit is None and potted_balls:
                self.first_ball_hit = potted_balls[0]
           
            # Check if cue ball is potted
            if not self.cue_ball.potted and self.cue_ball.in_hole():
                self.foul = True
                self.cue_ball.potted = True
                self.turn_ended = True
           
            if all_stopped or self.shot_motion_frames >= 900:
                if self.shot_motion_frames >= 900:
                    self.force_stop_balls()
                self.balls_moving = False
                self.shot_motion_frames = 0

                if self.winner is None and self.balls and all(ball.potted for ball in self.balls):
                    self.winner = self.current_player
               
                # Check if a valid shot was made
                if not self.foul and not self.valid_shot():
                    self.foul = True
                elif not self.foul and not self.turn_ended and not self.shot_potted_ball:
                    self.turn_ended = True
               
                if self.turn_ended or self.foul:
                    self.end_turn()
               
                # Check for game over condition
                if self.check_game_over():
                    self.state = GAME_OVER

    def any_ball_moving(self):
        for ball in [self.cue_ball] + self.balls:
            if not ball.potted and (abs(ball.vx) > 0.12 or abs(ball.vy) > 0.12):
                return True
        return False

    def settle_slow_balls(self):
        for ball in [self.cue_ball] + self.balls:
            if ball.potted:
                continue
            if abs(ball.vx) < 0.12:
                ball.vx = 0
            if abs(ball.vy) < 0.12:
                ball.vy = 0

    def force_stop_balls(self):
        for ball in [self.cue_ball] + self.balls:
            ball.vx = 0
            ball.vy = 0
   
    def check_first_collision(self):
        """Check for the first ball the cue ball hits"""
        cue_ball = self.cue_ball
        for ball in self.balls:
            if not ball.potted:
                dx = ball.x - cue_ball.x
                dy = ball.y - cue_ball.y
                distance = math.hypot(dx, dy)
                min_dist = cue_ball.radius + ball.radius
               
                if distance < min_dist:
                    return ball
        return None
   
    def valid_shot(self):
        """Check if the shot is valid according to 8-ball rules"""
        # If no ball was hit or potted
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
   
    def should_hit_8_ball(self):
        """Check if the player should be hitting the 8-ball"""
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
   
    def count_remaining_balls(self, player_index):
        """Count how many balls a player still needs to pot"""
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
   
    def process_potted_ball(self, ball):
        ball.potted = True
       
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
        # Reset cue ball if it was potted (ball in hand)
        if self.cue_ball.potted:
            self.cue_ball.x = WIDTH // 4
            self.cue_ball.y = HEIGHT // 2
            self.cue_ball.vx = 0
            self.cue_ball.vy = 0
            self.cue_ball.potted = False
       
        # Switch players if turn ended
        if self.turn_ended or self.foul:
            self.current_player = 1 - self.current_player
            self.foul = False
            self.turn_ended = False
            self.shot_potted_ball = False
            self.first_ball_hit = None
            if self.is_ai_turn():
                self.ai_shot_delay = 45
   
    def check_game_over(self):
        # Game is over if there's a winner
        if self.winner is None and self.balls and all(ball.potted for ball in self.balls):
            self.winner = self.current_player
        return self.winner is not None
   
    def handle_collisions(self):
        # Ball-to-Ball collisions
        collision_occurred = False
        for i in range(len(self.balls)):
            if self.balls[i].potted:
                continue
           
            # Check cue ball collision
            if not self.cue_ball.potted:
                if self.handle_ball_collision(self.cue_ball, self.balls[i]):
                    collision_occurred = True
           
            # Check other ball collisions
            for j in range(i + 1, len(self.balls)):
                if not self.balls[j].potted:
                    if self.handle_ball_collision(self.balls[i], self.balls[j]):
                        collision_occurred = True
        return collision_occurred
   
    def handle_ball_collision(self, ball1, ball2):
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

            play_sound("ball_collision")
            return True
        return False

    def toggle_graphics_style(self):
        self.graphics_style = "stylish" if self.graphics_style == "classic" else "classic"

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        flags = pygame.SCALED | (pygame.FULLSCREEN if self.is_fullscreen else 0)
        try:
            self.display_screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
        except pygame.error:
            fallback_flags = pygame.FULLSCREEN if self.is_fullscreen else 0
            self.display_screen = pygame.display.set_mode((WIDTH, HEIGHT), fallback_flags)

    def _draw_vertical_gradient(self, screen, top_color, bottom_color):
        for y in range(HEIGHT):
            t = y / max(1, HEIGHT - 1)
            color = (
                int(top_color[0] * (1 - t) + bottom_color[0] * t),
                int(top_color[1] * (1 - t) + bottom_color[1] * t),
                int(top_color[2] * (1 - t) + bottom_color[2] * t),
            )
            pygame.draw.line(screen, color, (0, y), (WIDTH, y))

    def _draw_centered_text(self, screen, text, font, color, center, shadow=True):
        rendered = font.render(text, True, color)
        rect = rendered.get_rect(center=center)
        if shadow:
            shadow_surf = font.render(text, True, (0, 0, 0))
            screen.blit(shadow_surf, (rect.x + 2, rect.y + 3))
        screen.blit(rendered, rect)
        return rect

    def _menu_label(self, option):
        if option == "Graphics Style":
            return f"Graphics Style: {self.graphics_style.title()}"
        if option == "Fullscreen":
            return f"Fullscreen: {'On' if self.is_fullscreen else 'Off'}"
        return option

    def _draw_glow_rect(self, screen, rect, fill, border, radius=12):
        glow = pygame.Surface((rect.width + 22, rect.height + 22), pygame.SRCALPHA)
        pygame.draw.rect(glow, (*border, 55), glow.get_rect(), border_radius=radius + 10)
        screen.blit(glow, (rect.x - 11, rect.y - 11))
        pygame.draw.rect(screen, fill, rect, border_radius=radius)
        pygame.draw.rect(screen, border, rect, 2, border_radius=radius)

    def _draw_switch(self, screen, rect, enabled):
        bg = (32, 163, 92) if enabled else (185, 54, 54)
        knob_x = rect.right - rect.height + 4 if enabled else rect.left + 4
        pygame.draw.rect(screen, bg, rect, border_radius=rect.height // 2)
        pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=rect.height // 2)
        pygame.draw.circle(screen, (245, 248, 246), (knob_x + rect.height // 2 - 4, rect.centery), rect.height // 2 - 6)
        label = "ON" if enabled else "OFF"
        text = game_font.render(label, True, WHITE)
        text_x = rect.left + 14 if enabled else rect.right - text.get_width() - 14
        screen.blit(text, (text_x, rect.centery - text.get_height() // 2))

    def _draw_dropdown(self, screen, key, rect, label, value_label, options):
        pygame.draw.rect(screen, (13, 25, 33), rect, border_radius=8)
        pygame.draw.rect(screen, (82, 173, 216), rect, 2, border_radius=8)
        caption = game_font.render(label, True, (184, 207, 215))
        value = score_font.render(value_label, True, WHITE)
        arrow = score_font.render("v" if self.training_dropdown_open != key else "^", True, (238, 196, 92))
        screen.blit(caption, (rect.x + 14, rect.y + 7))
        screen.blit(value, (rect.x + 14, rect.y + 30))
        screen.blit(arrow, (rect.right - 34, rect.y + 28))
        self.training_control_rects[key] = rect

        option_rects = []
        if self.training_dropdown_open == key:
            y = rect.bottom + 5
            for option_value, option_text in options:
                item = pygame.Rect(rect.x, y, rect.width, 42)
                pygame.draw.rect(screen, (18, 39, 50), item, border_radius=6)
                pygame.draw.rect(screen, (79, 141, 168), item, 1, border_radius=6)
                text = game_font.render(option_text, True, WHITE)
                screen.blit(text, (item.x + 14, item.y + 12))
                option_rects.append((option_value, item))
                y += 44
        self.training_control_rects[f"{key}_options"] = option_rects

    def _draw_segmented_control(self, screen, key, rect, label, options, selected_value):
        label_text = game_font.render(label, True, (218, 211, 185))
        screen.blit(label_text, (rect.x, rect.y - 26))

        gap = 6
        segment_w = (rect.width - gap * (len(options) - 1)) // len(options)
        option_rects = []
        for idx, (value, text) in enumerate(options):
            item = pygame.Rect(rect.x + idx * (segment_w + gap), rect.y, segment_w, rect.height)
            selected = value == selected_value
            fill = (198, 155, 72) if selected else (18, 35, 28)
            border = (255, 230, 158) if selected else (83, 118, 78)
            text_color = (23, 22, 17) if selected else (229, 231, 217)
            pygame.draw.rect(screen, (2, 5, 4), item.move(3, 4), border_radius=8)
            pygame.draw.rect(screen, fill, item, border_radius=8)
            pygame.draw.rect(screen, border, item, 2, border_radius=8)
            rendered = game_font.render(text, True, text_color)
            screen.blit(rendered, rendered.get_rect(center=item.center))
            option_rects.append((value, item))

        self.training_control_rects[f"{key}_options"] = option_rects

    def _draw_ball_slider(self, screen, rect):
        card = pygame.Rect(rect.x, rect.y - 28, rect.width, 70)
        pygame.draw.rect(screen, (18, 35, 28), card, border_radius=8)
        pygame.draw.rect(screen, (83, 118, 78), card, 2, border_radius=8)
        label = game_font.render("Ball Count", True, (218, 211, 185))
        count = score_font.render(str(self.training_ball_count), True, WHITE)
        screen.blit(label, (card.x + 14, card.y + 9))
        screen.blit(count, (card.right - count.get_width() - 14, card.y + 5))
        pygame.draw.line(screen, (67, 84, 62), (rect.left + 18, rect.centery), (rect.right - 18, rect.centery), 8)
        t = (self.training_ball_count - 2) / 13
        knob_x = int(rect.left + 18 + t * (rect.width - 36))
        pygame.draw.line(screen, (197, 155, 72), (rect.left + 18, rect.centery), (knob_x, rect.centery), 8)
        pygame.draw.circle(screen, (226, 182, 88), (knob_x, rect.centery), 15)
        pygame.draw.circle(screen, WHITE, (knob_x, rect.centery), 15, 2)
        self.training_control_rects["ball_slider"] = rect

    def _draw_menu_button(self, screen, rect, label, selected):
        if self.graphics_style == "classic":
            fill = (46, 55, 50) if not selected else (188, 54, 48)
            border = (154, 126, 72) if not selected else (255, 223, 142)
            text_color = (235, 232, 216) if not selected else WHITE
            shadow = (12, 10, 8)
            accent = (219, 181, 86)
        else:
            fill = (19, 36, 30) if not selected else (208, 166, 79)
            border = (88, 125, 86) if not selected else (255, 229, 155)
            text_color = (224, 232, 218) if not selected else (24, 21, 16)
            shadow = (4, 8, 7)
            accent = (170, 123, 58)

        pygame.draw.rect(screen, shadow, rect.move(5, 6), border_radius=8)
        pygame.draw.rect(screen, fill, rect, border_radius=8)
        pygame.draw.rect(screen, border, rect, 2 if not selected else 3, border_radius=8)

        if selected:
            marker = pygame.Rect(rect.left + 12, rect.centery - 9, 18, 18)
            pygame.draw.circle(screen, accent, marker.center, 9)
            pygame.draw.circle(screen, WHITE, marker.center, 9, 2)
        else:
            pygame.draw.line(screen, accent, (rect.left + 16, rect.bottom - 8), (rect.right - 16, rect.bottom - 8), 1)

        text = menu_font.render(label, True, text_color)
        text_rect = text.get_rect(center=rect.center)
        screen.blit(text, text_rect)
   
    def draw_menu(self, screen):
        if self.graphics_style == "classic":
            screen.blit(dark_wood_texture, (0, 0))
            table_rect = pygame.Rect(WIDTH // 2 - 400, 48, 800, 210)
            pygame.draw.rect(screen, (21, 13, 8), table_rect.move(7, 9), border_radius=28)
            pygame.draw.rect(screen, (104, 55, 24), table_rect, border_radius=28)
            pygame.draw.rect(screen, (205, 166, 94), table_rect, 4, border_radius=28)
            felt_rect = table_rect.inflate(-58, -58)
            pygame.draw.rect(screen, (18, 116, 70), felt_rect, border_radius=18)
            pygame.draw.rect(screen, (220, 196, 126), felt_rect, 2, border_radius=18)
            for hx, hy in [
                felt_rect.topleft, felt_rect.midtop, felt_rect.topright,
                felt_rect.bottomleft, felt_rect.midbottom, felt_rect.bottomright,
            ]:
                pygame.draw.circle(screen, (8, 8, 7), (int(hx), int(hy)), 18)
                pygame.draw.circle(screen, (223, 190, 114), (int(hx), int(hy)), 20, 3)
        else:
            self._draw_vertical_gradient(screen, (12, 17, 15), (31, 44, 34))
            for x in range(0, WIDTH, 120):
                pygame.draw.line(screen, (42, 57, 43), (x, 0), (x + 260, HEIGHT), 1)

            table_rect = pygame.Rect(WIDTH // 2 - 420, 56, 840, 260)
            pygame.draw.rect(screen, (3, 5, 4), table_rect.move(8, 10), border_radius=34)
            pygame.draw.rect(screen, (43, 30, 21), table_rect, border_radius=34)
            pygame.draw.rect(screen, (187, 150, 82), table_rect, 5, border_radius=34)
            rail_rect = table_rect.inflate(-36, -36)
            pygame.draw.rect(screen, (93, 45, 25), rail_rect, border_radius=24)
            pygame.draw.rect(screen, (220, 170, 87), rail_rect, 3, border_radius=24)
            felt_rect = table_rect.inflate(-70, -70)
            pygame.draw.rect(screen, (25, 103, 62), felt_rect, border_radius=18)
            pygame.draw.rect(screen, (161, 134, 72), felt_rect, 2, border_radius=18)
            pygame.draw.line(screen, (84, 151, 104), (felt_rect.left + 80, felt_rect.centery), (felt_rect.right - 80, felt_rect.centery), 2)
            for hx, hy in [
                felt_rect.topleft, felt_rect.midtop, felt_rect.topright,
                felt_rect.bottomleft, felt_rect.midbottom, felt_rect.bottomright,
            ]:
                pygame.draw.circle(screen, (0, 3, 8), (int(hx), int(hy)), 20)
                pygame.draw.circle(screen, (198, 159, 82), (int(hx), int(hy)), 24, 4)
            pygame.draw.circle(screen, (235, 235, 228), (felt_rect.left + 130, felt_rect.centery), 15)
            for idx, color in enumerate([(214, 42, 42), (246, 201, 67), (38, 116, 216), (27, 131, 65), (20, 20, 22)]):
                pygame.draw.circle(screen, color, (felt_rect.right - 210 + idx * 24, felt_rect.centery + (idx % 2) * 18 - 9), 12)

            for i in range(7):
                pygame.draw.circle(screen, (61, 54, 37), (WIDTH // 2 - 250 + i * 45, 38), 16, 3)
                pygame.draw.circle(screen, (61, 54, 37), (WIDTH // 2 + 250 - i * 45, 38), 16, 3)

        title_y = HEIGHT // 5
        panel = pygame.Rect(WIDTH // 2 - 305, HEIGHT // 2 - 160, 610, 405)
        panel_fill = (26, 31, 30) if self.graphics_style == "classic" else (14, 25, 21)
        panel_border = (201, 164, 89) if self.graphics_style == "classic" else (172, 132, 67)
        pygame.draw.rect(screen, (0, 0, 0), panel.move(7, 9), border_radius=14)
        pygame.draw.rect(screen, panel_fill, panel, border_radius=14)
        pygame.draw.rect(screen, panel_border, panel, 3, border_radius=14)

        self._draw_centered_text(screen, "8-Ball Pool", title_font, WHITE, (WIDTH // 2, title_y))
        subtitle_color = (235, 218, 172) if self.graphics_style == "classic" else (222, 216, 188)
        subtitle = game_font.render("Train, play, and style the table", True, subtitle_color)
        screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, title_y + 58))

        menu_top = HEIGHT // 2 - 115
        self.menu_item_rects = []
        for i, option in enumerate(self.menu_options):
            selected = i == self.selected_option
            label = self._menu_label(option)
            item_rect = pygame.Rect(WIDTH // 2 - 250, menu_top + i * 54 - 24, 500, 48)
            self.menu_item_rects.append(item_rect)
            self._draw_menu_button(screen, item_rect, label, selected)

        if self.ai_status_text:
            status_color = RED if "failed" in self.ai_status_text.lower() or "missing" in self.ai_status_text.lower() else LIGHT_GRAY
            status_text = game_font.render(self.ai_status_text, True, status_color)
            screen.blit(status_text, (WIDTH//2 - status_text.get_width()//2, HEIGHT - 70))
        elif self.graphics_style == "stylish":
            hint = game_font.render("Use LEFT/RIGHT or ENTER on style/fullscreen, or press F11 anytime", True, (188, 202, 196))
            screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT - 70))
   
    def draw_instructions(self, screen):
        if self.graphics_style == "classic":
            screen.blit(dark_wood_texture, (0, 0))
        else:
            self._draw_vertical_gradient(screen, (10, 15, 13), (29, 42, 32))
            for y in range(90, HEIGHT, 90):
                pygame.draw.line(screen, (43, 58, 43), (0, y), (WIDTH, y), 1)

        title_text = title_font.render("Instructions", True, WHITE)
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 70))
        sub = game_font.render("Play clean shots, claim your suit, finish on the 8-ball.", True, (222, 216, 188))
        screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 128))

        panel = pygame.Rect(WIDTH // 2 - 430, 168, 860, 474)
        pygame.draw.rect(screen, (0, 0, 0), panel.move(8, 10), border_radius=16)
        pygame.draw.rect(screen, (13, 25, 20), panel, border_radius=16)
        pygame.draw.rect(screen, (171, 132, 69), panel, 3, border_radius=16)

        instructions = [
            ("Aim", "Click and drag from the cue ball to set direction and power."),
            ("Turn", "You keep shooting only when you legally pot one of your balls."),
            ("Open Table", "The first legal solid or stripe you pot sets your target group."),
            ("Groups", "Solids are 1-7. Stripes are 9-15. The black 8-ball is last."),
            ("Finish", "Pot the 8-ball only after clearing your assigned group."),
            ("Fouls", "Cue-ball scratches, early 8-ball pots, or wrong first contact are punished."),
            ("Pause", "Press ESC during a match to pause."),
        ]

        left = panel.x + 54
        top = panel.y + 52
        row_h = 48
        for i, (label, body) in enumerate(instructions):
            y = top + i * row_h
            marker = pygame.Rect(left, y + 7, 82, 32)
            pygame.draw.rect(screen, (29, 52, 39), marker, border_radius=8)
            pygame.draw.rect(screen, (129, 103, 60), marker, 1, border_radius=8)
            label_text = game_font.render(label, True, (235, 215, 160))
            screen.blit(label_text, label_text.get_rect(center=marker.center))

            body_text = game_font.render(body, True, (232, 232, 220))
            screen.blit(body_text, (left + 108, y + 13))

        back_rect = pygame.Rect(WIDTH // 2 - 165, panel.bottom - 72, 330, 50)
        self.instruction_back_rect = back_rect
        pygame.draw.rect(screen, (4, 8, 6), back_rect.move(5, 6), border_radius=10)
        pygame.draw.rect(screen, (211, 166, 77), back_rect, border_radius=10)
        pygame.draw.rect(screen, (255, 232, 159), back_rect, 2, border_radius=10)
        back_text = menu_font.render("Back to Menu", True, (14, 24, 30))
        screen.blit(back_text, back_text.get_rect(center=back_rect.center))

        hint = game_font.render("Press ENTER or ESC to return.", True, (222, 216, 188))
        screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT - 90))
   
    def draw_paused(self, screen):
        # Semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (0, 0))
       
        # Draw paused text
        paused_text = title_font.render("GAME PAUSED", True, WHITE)
        screen.blit(paused_text, (WIDTH//2 - paused_text.get_width()//2, HEIGHT//2 - 50))
       
        # Draw options
        continue_text = menu_font.render("Press ESC to continue", True, LIGHT_GRAY)
        menu_text = menu_font.render("Press M to return to menu", True, LIGHT_GRAY)
       
        screen.blit(continue_text, (WIDTH//2 - continue_text.get_width()//2, HEIGHT//2 + 30))
        screen.blit(menu_text, (WIDTH//2 - menu_text.get_width()//2, HEIGHT//2 + 80))
   
    def draw_game_over(self, screen):
        # Semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (0, 0))
       
        # Draw game over message
        if self.winner is not None:
            winner_name = self.players[self.winner]["name"]
            winner_color = self.players[self.winner]["color"]
           
            game_over_text = title_font.render("GAME OVER", True, WHITE)
            winner_text = menu_font.render(f"{winner_name} Wins!", True, winner_color)
           
            screen.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, HEIGHT//2 - 70))
            screen.blit(winner_text, (WIDTH//2 - winner_text.get_width()//2, HEIGHT//2))
           
            # Draw return to menu message
            return_text = menu_font.render("Press ENTER to return to menu", True, LIGHT_GRAY)
            screen.blit(return_text, (WIDTH//2 - return_text.get_width()//2, HEIGHT//2 + 70))

    def _opponent_label(self, value):
        if value == "self":
            return "AI vs AI"
        if value == "none":
            return "Solo (No Opponent)"
        return "Random Agent"

    def _layout_label(self, value):
        if value == "beginner":
            return "Beginner"
        return "Rack"

    def _cycle_training_layout(self):
        return "rack" if self.training_layout == "beginner" else "beginner"

    def _cycle_opponent(self, direction):
        order = ["random", "self", "none"]
        idx = order.index(self.training_opponent)
        idx = (idx + direction) % len(order)
        return order[idx]

    def draw_training_menu(self, screen):
        self.training_control_rects = {}
        if self.graphics_style == "classic":
            screen.blit(dark_wood_texture, (0, 0))
        else:
            self._draw_vertical_gradient(screen, (10, 15, 13), (29, 42, 32))
            for y in range(90, HEIGHT, 90):
                pygame.draw.line(screen, (43, 58, 43), (0, y), (WIDTH, y), 1)

        title_text = title_font.render("Training Settings", True, WHITE)
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 70))
        sub = game_font.render("Build the agent from warmup drills to the full 15-ball match", True, (222, 216, 188))
        screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 128))

        back_rect = pygame.Rect(42, 42, 128, 46)
        self.training_control_rects["back"] = back_rect
        pygame.draw.rect(screen, (3, 6, 5), back_rect.move(4, 5), border_radius=8)
        pygame.draw.rect(screen, (22, 38, 30), back_rect, border_radius=8)
        pygame.draw.rect(screen, (169, 129, 68), back_rect, 2, border_radius=8)
        back_text = score_font.render("Back", True, (236, 230, 207))
        screen.blit(back_text, back_text.get_rect(center=back_rect.center))

        panel = pygame.Rect(WIDTH // 2 - 430, 168, 860, 474)
        pygame.draw.rect(screen, (0, 0, 0), panel.move(8, 10), border_radius=16)
        pygame.draw.rect(screen, (13, 25, 20), panel, border_radius=16)
        pygame.draw.rect(screen, (171, 132, 69), panel, 3, border_radius=16)

        left_x = panel.x + 46
        right_x = panel.centerx + 28
        row_y = panel.y + 82
        control_w = 350

        self._draw_ball_slider(screen, pygame.Rect(left_x, row_y + 22, control_w, 24))
        self._draw_segmented_control(
            screen,
            "layout",
            pygame.Rect(right_x, row_y + 8, control_w, 44),
            "Layout",
            [("beginner", "Beginner"), ("rack", "Rack")],
            self.training_layout
        )

        row_y += 116
        random_card = pygame.Rect(left_x, row_y - 10, control_w, 66)
        pygame.draw.rect(screen, (18, 35, 28), random_card, border_radius=8)
        pygame.draw.rect(screen, (83, 118, 78), random_card, 2, border_radius=8)
        random_label = score_font.render("Random Positions", True, WHITE)
        screen.blit(random_label, (random_card.x + 16, random_card.y + 18))
        random_rect = pygame.Rect(random_card.right - 112, random_card.y + 14, 94, 38)
        self._draw_switch(screen, random_rect, self.training_random_balls)
        self.training_control_rects["random"] = random_card

        self._draw_segmented_control(
            screen,
            "opponent",
            pygame.Rect(right_x, row_y + 8, control_w, 44),
            "Opponent",
            [("random", "Random"), ("self", "AI vs AI"), ("none", "Solo")],
            self.training_opponent
        )

        row_y += 116
        workout_card = pygame.Rect(left_x, row_y - 10, control_w, 66)
        pygame.draw.rect(screen, (18, 35, 28), workout_card, border_radius=8)
        pygame.draw.rect(screen, (83, 118, 78), workout_card, 2, border_radius=8)
        workout_label = score_font.render("Workout Plan", True, WHITE)
        screen.blit(workout_label, (workout_card.x + 16, workout_card.y + 18))
        workout_rect = pygame.Rect(workout_card.right - 112, workout_card.y + 14, 94, 38)
        self._draw_switch(screen, workout_rect, self.training_workout_plan)
        self.training_control_rects["workout"] = workout_card

        weird_card = pygame.Rect(right_x, row_y - 10, control_w, 66)
        pygame.draw.rect(screen, (18, 35, 28), weird_card, border_radius=8)
        pygame.draw.rect(screen, (83, 118, 78), weird_card, 2, border_radius=8)
        weird_label = score_font.render("Weird Graphics", True, WHITE)
        screen.blit(weird_label, (weird_card.x + 16, weird_card.y + 18))
        weird_rect = pygame.Rect(weird_card.right - 112, weird_card.y + 14, 94, 38)
        self._draw_switch(screen, weird_rect, self.training_weird_graphics)
        self.training_control_rects["weird"] = weird_card

        target = game_font.render(
            f"{self.training_ball_count} balls | {self._layout_label(self.training_layout)} | {self._opponent_label(self.training_opponent)}",
            True,
            (230, 226, 206)
        )
        screen.blit(target, (panel.centerx - target.get_width() // 2, panel.bottom - 100))

        start_rect = pygame.Rect(WIDTH // 2 - 160, panel.bottom - 72, 320, 52)
        self.training_control_rects["start"] = start_rect
        pygame.draw.rect(screen, (4, 8, 6), start_rect.move(5, 6), border_radius=10)
        pygame.draw.rect(screen, (211, 166, 77), start_rect, border_radius=10)
        pygame.draw.rect(screen, (255, 232, 159), start_rect, 2, border_radius=10)
        start_text = menu_font.render("Start Training", True, (14, 24, 30))
        screen.blit(start_text, start_text.get_rect(center=start_rect.center))

        help_text = game_font.render("Click any setting, drag ball count, or press ESC to return.", True, (222, 216, 188))
        screen.blit(help_text, (WIDTH//2 - help_text.get_width()//2, HEIGHT - 90))

    def draw_training(self, screen):
        if not self.training:
            return

        try:
            frame = self.training.get_frame()
        except Exception as exc:
            print(f"Training render error: {exc}")
            self.training.stop_with_error(exc)
            frame = None
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame)
            surf = pygame.transform.scale(surf, (WIDTH, HEIGHT))
            if self.training_weird_graphics:
                surf = self._make_weird_training_surface(surf)
            screen.blit(surf, (0, 0))

        overlay = pygame.Surface((WIDTH, 210), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        title = title_font.render("TRAINING MODE", True, WHITE)
        screen.blit(title, (20, 10))

        info_lines = self.training.get_status_lines()
        y = 60
        for line in info_lines:
            text = game_font.render(line, True, WHITE)
            screen.blit(text, (20, y))
            y += 20

    def _make_weird_training_surface(self, surf):
        tiny = pygame.transform.scale(surf, (max(1, WIDTH // 9), max(1, HEIGHT // 9)))
        weird = pygame.transform.scale(tiny, (WIDTH, HEIGHT))
        color_wash = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for y in range(0, HEIGHT, 28):
            band_color = (
                (255, 40 + (y * 3) % 180, 120, 54)
                if (y // 28) % 2 == 0
                else (55, 210, 255, 48)
            )
            pygame.draw.rect(color_wash, band_color, (0, y, WIDTH, 14))
        for x in range(-80, WIDTH, 130):
            pygame.draw.circle(color_wash, (235, 220, 55, 42), (x + 70, HEIGHT // 2), 95)
        weird.blit(color_wash, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        return weird

   
    def draw_table(self, screen):
        if self.graphics_style == "stylish":
            self.draw_stylish_table(screen)
            return

        # Draw the wooden border
        screen.blit(light_wood_texture, (0, 0))
       
        outer = pygame.Rect(54, 54, WIDTH - 108, HEIGHT - 108)
        rail = pygame.Rect(66, 66, WIDTH - 132, HEIGHT - 132)
        felt = pygame.Rect(80, 80, WIDTH - 160, HEIGHT - 160)

        pygame.draw.rect(screen, (30, 16, 10), outer.move(5, 7), border_radius=22)
        pygame.draw.rect(screen, (73, 35, 18), outer, border_radius=22)
        pygame.draw.rect(screen, (183, 139, 68), outer, 4, border_radius=22)
        pygame.draw.rect(screen, (107, 53, 25), rail, border_radius=16)
        pygame.draw.rect(screen, (220, 170, 88), rail, 2, border_radius=16)
        pygame.draw.rect(screen, (18, 104, 56), felt, border_radius=8)
        pygame.draw.rect(screen, (10, 68, 39), felt.inflate(-20, -20), 1, border_radius=6)
       
        # Draw cushions (with 3D effect)
        cushion_depth = 10
        cushion_color = (14, 82, 45)
       
        # Top cushion
        pygame.draw.rect(screen, cushion_color, (80, 70, WIDTH-160, cushion_depth), border_radius=4)
        # Bottom cushion
        pygame.draw.rect(screen, cushion_color, (80, HEIGHT-80, WIDTH-160, cushion_depth), border_radius=4)
        # Left cushion
        pygame.draw.rect(screen, cushion_color, (70, 80, cushion_depth, HEIGHT-160), border_radius=4)
        # Right cushion
        pygame.draw.rect(screen, cushion_color, (WIDTH-80, 80, cushion_depth, HEIGHT-160), border_radius=4)
       
       
       
        # Draw pockets/holes with nice 3D effect
        for hx, hy in holes:
            # Draw pocket shadow
            pygame.draw.circle(screen, (0, 0, 0), (hx+2, hy+2), HOLE_RADIUS+2)
            # Draw pocket hole
            pygame.draw.circle(screen, (10, 10, 10), (hx, hy), HOLE_RADIUS)
            # Draw pocket rim
            pygame.draw.circle(screen, (178, 133, 67), (hx, hy), HOLE_RADIUS+5, 5)
            pygame.draw.circle(screen, (44, 24, 14), (hx, hy), HOLE_RADIUS+1, 2)

    def draw_stylish_table(self, screen):
        self._draw_vertical_gradient(screen, (9, 13, 12), (28, 38, 30))

        shadow = pygame.Rect(42, 54, WIDTH - 84, HEIGHT - 94)
        pygame.draw.rect(screen, (0, 0, 0), shadow, border_radius=30)

        outer = pygame.Rect(48, 48, WIDTH - 96, HEIGHT - 96)
        rail = pygame.Rect(58, 62, WIDTH - 116, HEIGHT - 124)
        felt = pygame.Rect(92, 92, WIDTH - 184, HEIGHT - 184)

        pygame.draw.rect(screen, (31, 22, 17), outer, border_radius=28)
        pygame.draw.rect(screen, (178, 139, 72), outer, 5, border_radius=28)
        pygame.draw.rect(screen, (78, 38, 24), rail, border_radius=22)

        top_rail = pygame.Rect(rail.left + 24, rail.top + 8, rail.width - 48, felt.top - rail.top - 8)
        bottom_rail = pygame.Rect(rail.left + 24, felt.bottom, rail.width - 48, rail.bottom - felt.bottom - 8)
        left_rail = pygame.Rect(rail.left + 8, rail.top + 24, felt.left - rail.left - 8, rail.height - 48)
        right_rail = pygame.Rect(felt.right, rail.top + 24, rail.right - felt.right - 8, rail.height - 48)
        for band in (top_rail, bottom_rail, left_rail, right_rail):
            pygame.draw.rect(screen, (122, 66, 37), band, border_radius=8)
            pygame.draw.rect(screen, (214, 165, 86), band, 2, border_radius=8)

        pygame.draw.rect(screen, (21, 89, 54), felt, border_radius=12)
        pygame.draw.rect(screen, (153, 122, 65), felt, 2, border_radius=12)

        # Subtle felt stripes.
        stripe_w = max(24, felt.width // 18)
        for i, x in enumerate(range(felt.left, felt.right, stripe_w)):
            color = (22, 92, 56) if i % 2 == 0 else (25, 103, 61)
            pygame.draw.rect(screen, color, (x, felt.top, stripe_w, felt.height))
        pygame.draw.line(screen, (88, 143, 96), (felt.left + 120, felt.centery), (felt.right - 120, felt.centery), 2)
        pygame.draw.rect(screen, (164, 131, 72), felt, 2, border_radius=12)
        pygame.draw.rect(screen, (10, 47, 30), felt.inflate(-18, -18), 1, border_radius=10)

        # Repaint the side rails after the felt so both sides read as wood, not felt.
        for side_rail in (left_rail, right_rail):
            pygame.draw.rect(screen, (90, 43, 25), side_rail, border_radius=8)
            pygame.draw.rect(screen, (137, 75, 41), side_rail.inflate(-10, -6), border_radius=6)
            pygame.draw.rect(screen, (218, 169, 88), side_rail, 2, border_radius=8)
            inner_trim_x = side_rail.right - 4 if side_rail is left_rail else side_rail.left + 3
            pygame.draw.line(screen, (235, 196, 116), (inner_trim_x, side_rail.top + 18), (inner_trim_x, side_rail.bottom - 18), 2)

        # Rail diamonds.
        diamond_color = (223, 192, 119)
        for x in range(felt.left + felt.width // 6, felt.right, felt.width // 6):
            pygame.draw.polygon(screen, diamond_color, [(x, top_rail.centery - 6), (x + 6, top_rail.centery), (x, top_rail.centery + 6), (x - 6, top_rail.centery)])
            pygame.draw.polygon(screen, diamond_color, [(x, bottom_rail.centery - 6), (x + 6, bottom_rail.centery), (x, bottom_rail.centery + 6), (x - 6, bottom_rail.centery)])
        for y in range(felt.top + felt.height // 4, felt.bottom, felt.height // 4):
            pygame.draw.polygon(screen, diamond_color, [(left_rail.centerx, y - 6), (left_rail.centerx + 6, y), (left_rail.centerx, y + 6), (left_rail.centerx - 6, y)])
            pygame.draw.polygon(screen, diamond_color, [(right_rail.centerx, y - 6), (right_rail.centerx + 6, y), (right_rail.centerx, y + 6), (right_rail.centerx - 6, y)])

        for hx, hy in holes:
            pygame.draw.circle(screen, (1, 5, 7), (hx + 4, hy + 5), HOLE_RADIUS + 8)
            pygame.draw.circle(screen, (3, 10, 12), (hx, hy), HOLE_RADIUS + 4)
            pygame.draw.circle(screen, (190, 151, 76), (hx, hy), HOLE_RADIUS + 8, 4)
            pygame.draw.circle(screen, (32, 19, 13), (hx, hy), HOLE_RADIUS + 3, 2)
   
    def draw_ball_counters(self, screen):
    # Draw counters for each player's remaining balls
        for i, player in enumerate(self.players):
            # Determine position based on player
            panel_w, panel_h = 170, 54
            x_pos = 96 if i == 0 else WIDTH - panel_w - 96
            y_pos = 8
           
            # Draw player info background
            panel = pygame.Rect(x_pos, y_pos, panel_w, panel_h)
            pygame.draw.rect(screen, (3, 6, 5), panel.move(3, 4), border_radius=8)
            pygame.draw.rect(screen, (34, 31, 25), panel, 0, border_radius=8)
           
            # Highlight current player with thicker border
            border_color = (203, 160, 82) if i == self.current_player else (107, 95, 64)
            pygame.draw.rect(screen, border_color, panel, 2, border_radius=8)
            pygame.draw.rect(screen, player["color"], (x_pos + 8, y_pos + 8, 5, panel_h - 16), border_radius=3)
           
            # Draw player name with border
            name_text = player["name"]
            name_rendered = game_font.render(name_text, True, WHITE)
            name_width = name_rendered.get_width()
            text_x = x_pos + 24
            text_y = y_pos + 7
           
            # Draw border by rendering multiple offset versions
            border_thickness = 2
            for off_x in range(-border_thickness, border_thickness+1):
                for off_y in range(-border_thickness, border_thickness+1):
                    if off_x != 0 or off_y != 0:  # Skip the center
                        border_render = game_font.render(name_text, True, BLACK)
                        screen.blit(border_render, (text_x + off_x, text_y + off_y))
           
            # Draw the main text on top
            screen.blit(name_rendered, (text_x, text_y))
           
            # Draw ball type (show ??? if not assigned yet)
            ball_type = player["type"]
            type_text = ball_type.capitalize() if ball_type else "Open"
            type_rendered = game_font.render(type_text, True, WHITE)
            screen.blit(type_rendered, (x_pos + 24, y_pos + 30))
           
            # Draw remaining balls (show 7 if not assigned yet)
            remaining = self.count_remaining_balls(i)
            remain_text = game_font.render(f"Left {remaining}", True, WHITE)
            screen.blit(remain_text, (x_pos + panel_w - remain_text.get_width() - 14, y_pos + 30))
 
    def draw_stylish_ball(self, screen, ball):
        x, y, r = int(ball.x), int(ball.y), int(ball.radius)
        pygame.draw.circle(screen, (0, 0, 0), (x + 4, y + 5), r + 2)
        pygame.draw.circle(screen, ball.color, (x, y), r)
        pygame.draw.circle(screen, (255, 255, 255), (x - r // 3, y - r // 3), max(3, r // 3))
        pygame.draw.circle(screen, (255, 255, 255), (x, y), r, 1)
        if ball.number > 0:
            label_color = BLACK if ball.color in (WHITE, YELLOW, ORANGE) else WHITE
            number_font = pygame.font.SysFont('Arial', 10, bold=True)
            number_text = number_font.render(str(ball.number), True, label_color)
            screen.blit(number_text, number_text.get_rect(center=(x, y)))

    def draw_stylish_ball_counters(self, screen):
        panel_w, panel_h = 174, 52
        positions = [(98, 8), (WIDTH - panel_w - 98, 8)]
        for i, player in enumerate(self.players):
            x, y = positions[i]
            selected = i == self.current_player
            panel = pygame.Rect(x, y, panel_w, panel_h)
            pygame.draw.rect(screen, (3, 6, 5), panel.move(3, 4), border_radius=8)
            pygame.draw.rect(screen, (17, 28, 22), panel, border_radius=8)
            pygame.draw.rect(screen, (203, 160, 82) if selected else (107, 95, 64), panel, 2, border_radius=8)
            accent = pygame.Rect(panel.x + 8, panel.y + 8, 5, panel.height - 16)
            pygame.draw.rect(screen, player["color"], accent, border_radius=3)

            name_text = game_font.render(player["name"], True, WHITE)
            screen.blit(name_text, (x + 20, y + 7))
            ball_type = player["type"].capitalize() if player["type"] else "Open"
            meta = game_font.render(f"{ball_type} | Left {self.count_remaining_balls(i)}", True, (224, 218, 194))
            screen.blit(meta, (x + 20, y + 29))
    def draw_game(self, screen):
        # Draw table and elements
        self.draw_table(screen)
       
        # Draw balls
        for ball in self.balls:
            if not ball.potted:
                if self.graphics_style == "stylish":
                    self.draw_stylish_ball(screen, ball)
                else:
                    ball.draw(screen)
       
        if not self.cue_ball.potted:
            if self.graphics_style == "stylish":
                self.draw_stylish_ball(screen, self.cue_ball)
            else:
                self.cue_ball.draw(screen)
       
        # Draw cue when a human player is aiming
        if not self.balls_moving and not self.cue_ball.potted and not self.is_ai_turn():
            self.cue.draw(screen, self.cue_ball.x, self.cue_ball.y, self.balls)
       
        # Draw ball counters
        if self.graphics_style == "stylish":
            self.draw_stylish_ball_counters(screen)
        else:
            self.draw_ball_counters(screen)
       
        # Draw current player indicator
        current_player = self.players[self.current_player]
        indicator_text = score_font.render(f"Current Turn: {current_player['name']}", True, current_player["color"] if self.graphics_style == "classic" else WHITE)
        if self.graphics_style == "stylish":
            pill = pygame.Rect(WIDTH // 2 - 145, 10, 290, 36)
            pygame.draw.rect(screen, (10, 18, 15), pill, border_radius=8)
            pygame.draw.rect(screen, current_player["color"], pill, 2, border_radius=8)
            screen.blit(indicator_text, indicator_text.get_rect(center=pill.center))
        else:
            screen.blit(indicator_text, (WIDTH // 2 - indicator_text.get_width() // 2, 20))
       
        # Draw game status messages if applicable
        if self.foul:
            foul_text = game_font.render("FOUL! Click to continue", True, RED)
            screen.blit(foul_text, (WIDTH//2 - foul_text.get_width()//2, HEIGHT - 40))
           
        if self.ball_assignment_done and self.should_hit_8_ball():
            eight_text = game_font.render("Target: 8-Ball", True, WHITE)
            screen.blit(eight_text, (WIDTH//2 - eight_text.get_width()//2, HEIGHT - 60))

        if self.is_ai_turn() and not self.balls_moving:
            ai_text = game_font.render("Trained AI is aiming...", True, LIGHT_GRAY)
            screen.blit(ai_text, (WIDTH//2 - ai_text.get_width()//2, HEIGHT - 40))
        elif self.vs_trained_agent and self.ai_status_text:
            status_color = RED if "failed" in self.ai_status_text.lower() else LIGHT_GRAY
            status_text = game_font.render(self.ai_status_text, True, status_color)
            screen.blit(status_text, (WIDTH//2 - status_text.get_width()//2, HEIGHT - 40))
   
    def run(self, screen):
        running = True
        self.display_screen = screen
        while running:
            if not pygame.get_init() or not pygame.display.get_init():
                break

            try:
                events = pygame.event.get()
            except Exception as exc:
                print(f"Pygame event error: {exc}")
                break

            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                    screen = self.display_screen
                    continue
               
                # Handle events based on current state
                if self.state == MENU:
                    running = self.handle_menu_events(event)
                elif self.state == GAME:
                    running = self.handle_game_events(event)
                elif self.state == PAUSED:
                    running = self.handle_paused_events(event)
                elif self.state == INSTRUCTIONS:
                    running = self.handle_instructions_events(event)
                elif self.state == GAME_OVER:
                    running = self.handle_game_over_events(event)
                elif self.state == TRAINING:
                    running = self.handle_training_events(event)
                elif self.state == TRAINING_MENU:
                    running = self.handle_training_menu_events(event)

                screen = self.display_screen
                
                if not running:
                    break

            if not running:
                break
           
            # Update game logic if in game state
            if self.state == GAME:
                self.update_game()
            elif self.state == TRAINING and self.training:
                try:
                    self.training.step()
                except Exception as exc:
                    print(f"Training error: {exc}")
                    self.training.stop_with_error(exc)
           
            if not pygame.display.get_init():
                break

            # Clear screen
            try:
                screen.fill(BLACK)
            except pygame.error as exc:
                print(f"Display error: {exc}")
                break
           
            # Draw based on current state
            if self.state == MENU:
                self.draw_menu(screen)
            elif self.state == GAME:
                self.draw_game(screen)
            elif self.state == PAUSED:
                self.draw_game(screen)
                self.draw_paused(screen)
            elif self.state == INSTRUCTIONS:
                self.draw_instructions(screen)
            elif self.state == GAME_OVER:
                self.draw_game(screen)
                self.draw_game_over(screen)
            elif self.state == TRAINING:
                self.draw_training(screen)
            elif self.state == TRAINING_MENU:
                self.draw_training_menu(screen)
           
            # Update display
            try:
                pygame.display.flip()
            except pygame.error as exc:
                print(f"Display error: {exc}")
                break
            clock.tick(60)
       
        training = self.training
        self.training = None
        pygame.quit()
        if training:
            training.close()
