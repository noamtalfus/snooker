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
        self.active_ball_count = 15
        self.graphics_style = "classic"
        self.reset_game()
        self.state = MENU
        self.menu_options = ["New Game", "Play vs Trained AI", "Training Mode", "Instructions", "Graphics Style", "Quit"]
        self.selected_option = 0
        self.training = None
        self.training_selected_option = 0
        self.training_ball_count = 15
        self.training_layout = "rack"
        self.training_random_balls = False
        self.training_opponent = "random"  # random | self | none
        self.training_workout_plan = True
        self.training_dropdown_open = None
        self.training_control_rects = {}
        self.menu_item_rects = []
       
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
        self.balls_moving = False
        self.last_potted = None
        self.foul = False
        self.ball_assignment_done = False  # Track if balls have been assigned to players
        self.target_ball = None  # 8-ball in the final stage
        self.first_ball_hit = None
        self.winner = None
       
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
                self.first_ball_hit = None




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

        self.ai_status_text = f"Playing against trained AI ({self.trained_agent.ball_count} balls)"
        self.reset_game(self.trained_agent.ball_count, True)
        self.state = GAME

    def start_training(self, ball_count, random_balls, opponent_type, layout, workout_plan):
        self.close_training()
        self.training = TrainingController(
            ball_count=ball_count,
            random_balls=random_balls,
            opponent_type=opponent_type,
            resume_checkpoint=False,
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

            slider_rect = self.training_control_rects.get("ball_slider")
            if slider_rect and slider_rect.collidepoint(mouse_pos):
                t = (mouse_pos[0] - slider_rect.left) / max(1, slider_rect.width)
                self.training_ball_count = max(2, min(15, int(round(2 + t * 13))))
                return True

            for key in ("random", "workout"):
                rect = self.training_control_rects.get(key)
                if rect and rect.collidepoint(mouse_pos):
                    if key == "random":
                        self.training_random_balls = not self.training_random_balls
                    else:
                        self.training_workout_plan = not self.training_workout_plan
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
                self.training_selected_option = (self.training_selected_option - 1) % 6
            elif event.key == pygame.K_DOWN:
                self.training_selected_option = (self.training_selected_option + 1) % 6
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
            elif event.key == pygame.K_RETURN:
                if self.training_selected_option == 5:
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
        self.first_ball_hit = None
        self.ai_shot_delay = 45
   
    def update_game(self):
        if self.is_ai_turn() and not self.balls_moving and not self.turn_ended:
            self.take_ai_shot()

        if self.balls_moving:
            
            collided_balls = self.check_first_collision()
            if collided_balls and not self.first_ball_hit:
                self.first_ball_hit = collided_balls
           
            
            all_stopped = True
            for ball in [self.cue_ball] + self.balls:
                if not ball.potted:
                    ball.move()
                    if abs(ball.vx) > 0.1 or abs(ball.vy) > 0.1:
                        all_stopped = False
           
            collision_occurred = self.handle_collisions()
            if collision_occurred:
                all_stopped = False
           
            
            potted_balls = []
            for ball in self.balls:
                if not ball.potted and ball.in_hole():
                    potted_balls.append(ball)
           
            # Process all potted balls
            for ball in potted_balls:
                self.process_potted_ball(ball)

            if self.first_ball_hit is None and potted_balls:
                self.first_ball_hit = potted_balls[0]
           
            # Check if cue ball is potted
            if not self.cue_ball.potted and self.cue_ball.in_hole():
                self.foul = True
                self.cue_ball.potted = True
                self.turn_ended = True
           
            if all_stopped:
                self.balls_moving = False

                if self.winner is None and self.balls and all(ball.potted for ball in self.balls):
                    self.winner = self.current_player
               
                # Check if a valid shot was made
                if not self.foul and not self.valid_shot():
                    self.foul = True
               
                if self.turn_ended or self.foul:
                    self.end_turn()
               
                # Check for game over condition
                if self.check_game_over():
                    self.state = GAME_OVER
   
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
            # Collision detected - calculate collision response
            angle = math.atan2(dy, dx)
           
            # Move balls apart to prevent sticking
            overlap = min_dist - distance
            ball1.x -= overlap * math.cos(angle) / 2
            ball1.y -= overlap * math.sin(angle) / 2
            ball2.x += overlap * math.cos(angle) / 2
            ball2.y += overlap * math.sin(angle) / 2
           
            # Calculate new velocities using conservation of momentum
            v1 = math.hypot(ball1.vx, ball1.vy)
            v2 = math.hypot(ball2.vx, ball2.vy)
           
            dir1 = math.atan2(ball1.vy, ball1.vx) if v1 > 0 else 0
            dir2 = math.atan2(ball2.vy, ball2.vx) if v2 > 0 else 0
           
            # Compute new velocities (simplified physics)
            new_x_vel1 = v2 * math.cos(dir2 - angle) * math.cos(angle)
            new_y_vel1 = v2 * math.cos(dir2 - angle) * math.sin(angle)
            new_x_vel2 = v1 * math.cos(dir1 - angle) * math.cos(angle)
            new_y_vel2 = v1 * math.cos(dir1 - angle) * math.sin(angle)
           
            # Update velocities
            ball1.vx = new_x_vel1 * 0.95  # Small damping factor
            ball1.vy = new_y_vel1 * 0.95
            ball2.vx = new_x_vel2 * 0.95
            ball2.vy = new_y_vel2 * 0.95
           
            # Add tangential velocities back
            ball1.vx += v1 * math.sin(dir1 - angle) * math.cos(angle + math.pi/2) * 0.95
            ball1.vy += v1 * math.sin(dir1 - angle) * math.sin(angle + math.pi/2) * 0.95
            ball2.vx += v2 * math.sin(dir2 - angle) * math.cos(angle + math.pi/2) * 0.95
            ball2.vy += v2 * math.sin(dir2 - angle) * math.sin(angle + math.pi/2) * 0.95
           
            play_sound("ball_collision")
            return True
        return False

    def toggle_graphics_style(self):
        self.graphics_style = "stylish" if self.graphics_style == "classic" else "classic"

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

    def _draw_ball_slider(self, screen, rect):
        pygame.draw.rect(screen, (13, 25, 33), rect.inflate(0, 28), border_radius=8)
        pygame.draw.rect(screen, (82, 173, 216), rect.inflate(0, 28), 2, border_radius=8)
        label = game_font.render("Ball Count", True, (184, 207, 215))
        count = score_font.render(str(self.training_ball_count), True, WHITE)
        screen.blit(label, (rect.x + 14, rect.y - 18))
        screen.blit(count, (rect.right - count.get_width() - 14, rect.y - 22))
        pygame.draw.line(screen, (74, 105, 116), (rect.left + 18, rect.centery), (rect.right - 18, rect.centery), 8)
        t = (self.training_ball_count - 2) / 13
        knob_x = int(rect.left + 18 + t * (rect.width - 36))
        pygame.draw.line(screen, (80, 190, 230), (rect.left + 18, rect.centery), (knob_x, rect.centery), 8)
        pygame.draw.circle(screen, (241, 198, 90), (knob_x, rect.centery), 15)
        pygame.draw.circle(screen, WHITE, (knob_x, rect.centery), 15, 2)
        self.training_control_rects["ball_slider"] = rect
   
    def draw_menu(self, screen):
        if self.graphics_style == "classic":
            screen.blit(dark_wood_texture, (0, 0))
        else:
            self._draw_vertical_gradient(screen, (5, 15, 31), (8, 50, 78))
            for x in range(0, WIDTH, 120):
                pygame.draw.line(screen, (12, 58, 82), (x, 0), (x + 260, HEIGHT), 1)

            table_rect = pygame.Rect(WIDTH // 2 - 420, 56, 840, 260)
            pygame.draw.rect(screen, (2, 8, 15), table_rect.move(8, 10), border_radius=34)
            pygame.draw.rect(screen, (34, 48, 58), table_rect, border_radius=34)
            pygame.draw.rect(screen, (207, 214, 214), table_rect, 5, border_radius=34)
            rail_rect = table_rect.inflate(-36, -36)
            pygame.draw.rect(screen, (151, 64, 34), rail_rect, border_radius=24)
            pygame.draw.rect(screen, (233, 128, 71), rail_rect, 3, border_radius=24)
            felt_rect = table_rect.inflate(-70, -70)
            pygame.draw.rect(screen, (25, 163, 220), felt_rect, border_radius=18)
            pygame.draw.rect(screen, (88, 219, 255), felt_rect, 2, border_radius=18)
            pygame.draw.line(screen, (134, 226, 252), (felt_rect.left + 80, felt_rect.centery), (felt_rect.right - 80, felt_rect.centery), 2)
            for hx, hy in [
                felt_rect.topleft, felt_rect.midtop, felt_rect.topright,
                felt_rect.bottomleft, felt_rect.midbottom, felt_rect.bottomright,
            ]:
                pygame.draw.circle(screen, (0, 3, 8), (int(hx), int(hy)), 20)
                pygame.draw.circle(screen, (221, 225, 222), (int(hx), int(hy)), 24, 4)
            pygame.draw.circle(screen, (235, 235, 228), (felt_rect.centerx - 90, felt_rect.centery), 15)
            for idx, color in enumerate([(214, 42, 42), (246, 201, 67), (38, 116, 216), (27, 131, 65), (20, 20, 22)]):
                pygame.draw.circle(screen, color, (felt_rect.centerx + 90 + idx * 24, felt_rect.centery + (idx % 2) * 18 - 9), 12)

            for side_x, name, color in [(54, "ZEE", (238, 185, 61)), (WIDTH - 122, "OREX", (70, 144, 220))]:
                avatar = pygame.Rect(side_x, 34, 68, 68)
                pygame.draw.rect(screen, color, avatar, border_radius=10)
                pygame.draw.rect(screen, WHITE, avatar, 2, border_radius=10)
                pygame.draw.circle(screen, (20, 23, 28), avatar.center, 22)
                text = game_font.render(name, True, WHITE)
                screen.blit(text, (avatar.centerx - text.get_width() // 2, avatar.bottom + 8))

            for i in range(7):
                pygame.draw.circle(screen, (17, 42, 68), (WIDTH // 2 - 250 + i * 45, 38), 16, 3)
                pygame.draw.circle(screen, (17, 42, 68), (WIDTH // 2 + 250 - i * 45, 38), 16, 3)

        title_y = HEIGHT // 5
        self._draw_centered_text(screen, "8-Ball Pool", title_font, WHITE, (WIDTH // 2, title_y))
        subtitle = game_font.render("Train, play, and style the table", True, (210, 220, 214))
        screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, title_y + 58))

        menu_top = HEIGHT // 2 - 115
        self.menu_item_rects = []
        for i, option in enumerate(self.menu_options):
            selected = i == self.selected_option
            label = self._menu_label(option)
            color = WHITE if selected else LIGHT_GRAY
            text = menu_font.render(label, True, color)
            text_rect = text.get_rect(center=(WIDTH//2, menu_top + i * 54))
            item_rect = pygame.Rect(WIDTH // 2 - 250, text_rect.centery - 24, 500, 48)
            self.menu_item_rects.append(item_rect)

            if i == self.selected_option:
                if self.graphics_style == "classic":
                    pygame.draw.rect(screen, PLAYER1_COLOR,
                                   (text_rect.left - 20, text_rect.top - 5,
                                    text_rect.width + 40, text_rect.height + 10),
                                   3, border_radius=10)
                else:
                    card = item_rect
                    pygame.draw.rect(screen, (237, 188, 92), card, border_radius=8)
                    pygame.draw.rect(screen, (255, 238, 178), card, 2, border_radius=8)
                    text = menu_font.render(label, True, (18, 27, 31))
                    text_rect = text.get_rect(center=card.center)
            elif self.graphics_style == "stylish":
                pygame.draw.rect(screen, (8, 25, 42), item_rect, border_radius=8)
                pygame.draw.rect(screen, (40, 85, 110), item_rect, 1, border_radius=8)

            screen.blit(text, text_rect)

        if self.ai_status_text:
            status_color = RED if "failed" in self.ai_status_text.lower() or "missing" in self.ai_status_text.lower() else LIGHT_GRAY
            status_text = game_font.render(self.ai_status_text, True, status_color)
            screen.blit(status_text, (WIDTH//2 - status_text.get_width()//2, HEIGHT - 70))
        elif self.graphics_style == "stylish":
            hint = game_font.render("Use LEFT/RIGHT or ENTER on Graphics Style to switch looks", True, (188, 202, 196))
            screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT - 70))
   
    def draw_instructions(self, screen):
        screen.blit(dark_wood_texture, (0, 0))
       
        # Draw title
        title_text = title_font.render("Instructions", True, WHITE)
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 50))
       
        # Draw instruction text
        instructions = [
            "- Click and drag to aim and set power for your shot",
            "- First player to legally pot all their balls and then the 8-ball wins",
            "- Solid balls are numbered 1-7, Striped balls are 9-15",
            "- The 8-ball is the black ball numbered 8",
            "- After potting a ball of a certain type (solid/striped), that type becomes your target",
            "- Pot the 8-ball only after all your assigned balls are potted",
            "- Potting the cue ball (white) or the 8-ball too early results in a foul",
            "- Press ESC to pause the game"
        ]
       
        y_offset = 150
        for line in instructions:
            text = game_font.render(line, True, WHITE)
            screen.blit(text, (WIDTH//2 - text.get_width()//2, y_offset))
            y_offset += 40
           
        # Draw return message
        back_text = menu_font.render("Press ENTER or ESC to return to menu", True, LIGHT_GRAY)
        screen.blit(back_text, (WIDTH//2 - back_text.get_width()//2, HEIGHT - 100))
   
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
            self._draw_vertical_gradient(screen, (5, 16, 32), (7, 56, 82))
            for y in range(90, HEIGHT, 90):
                pygame.draw.line(screen, (15, 66, 93), (0, y), (WIDTH, y), 1)

        title_text = title_font.render("Training Settings", True, WHITE)
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 70))
        sub = game_font.render("Build the agent from warmup drills to the full 15-ball match", True, (205, 221, 224))
        screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 128))

        panel = pygame.Rect(WIDTH // 2 - 390, 170, 780, 455)
        self._draw_glow_rect(screen, panel, (8, 22, 32), (73, 167, 208), 16)

        left_x = panel.x + 46
        right_x = panel.centerx + 28
        row_y = panel.y + 76
        control_w = 300

        self._draw_ball_slider(screen, pygame.Rect(left_x, row_y + 22, control_w, 24))
        self._draw_dropdown(
            screen,
            "layout",
            pygame.Rect(right_x, row_y - 12, control_w, 68),
            "Layout",
            self._layout_label(self.training_layout),
            [("beginner", "Beginner"), ("rack", "Rack")]
        )

        row_y += 112
        random_label = score_font.render("Random Positions", True, WHITE)
        screen.blit(random_label, (left_x, row_y + 8))
        random_rect = pygame.Rect(left_x + 205, row_y + 2, 94, 38)
        self._draw_switch(screen, random_rect, self.training_random_balls)
        self.training_control_rects["random"] = random_rect

        self._draw_dropdown(
            screen,
            "opponent",
            pygame.Rect(right_x, row_y - 12, control_w, 68),
            "Opponent",
            self._opponent_label(self.training_opponent),
            [("random", "Random Agent"), ("self", "AI vs AI"), ("none", "Solo")]
        )

        row_y += 112
        workout_label = score_font.render("Workout Plan", True, WHITE)
        screen.blit(workout_label, (left_x, row_y + 8))
        workout_rect = pygame.Rect(left_x + 205, row_y + 2, 94, 38)
        self._draw_switch(screen, workout_rect, self.training_workout_plan)
        self.training_control_rects["workout"] = workout_rect

        target = game_font.render(
            f"Target: {self.training_ball_count} balls | {self._layout_label(self.training_layout)} | {self._opponent_label(self.training_opponent)}",
            True,
            (205, 221, 224)
        )
        screen.blit(target, (right_x, row_y + 14))

        start_rect = pygame.Rect(WIDTH // 2 - 160, panel.bottom - 72, 320, 52)
        self.training_control_rects["start"] = start_rect
        pygame.draw.rect(screen, (235, 180, 72), start_rect, border_radius=10)
        pygame.draw.rect(screen, (255, 239, 181), start_rect, 2, border_radius=10)
        start_text = menu_font.render("Start Training", True, (14, 24, 30))
        screen.blit(start_text, start_text.get_rect(center=start_rect.center))

        help_text = game_font.render("Click controls or use keyboard. ESC returns to menu.", True, (205, 221, 224))
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

   
    def draw_table(self, screen):
        if self.graphics_style == "stylish":
            self.draw_stylish_table(screen)
            return

        # Draw the wooden border
        screen.blit(light_wood_texture, (0, 0))
       
        # Draw outer and inner table edges for a 3D effect
        pygame.draw.rect(screen, DARK_WOOD, (60, 60, WIDTH-120, HEIGHT-120), 0)
        pygame.draw.rect(screen, TABLE_COLOR, (80, 80, WIDTH-160, HEIGHT-160), 0)
       
        # Draw cushions (with 3D effect)
        cushion_depth = 10
        cushion_color = (0, 100, 0)
       
        # Top cushion
        pygame.draw.rect(screen, cushion_color, (80, 70, WIDTH-160, cushion_depth))
        # Bottom cushion
        pygame.draw.rect(screen, cushion_color, (80, HEIGHT-80, WIDTH-160, cushion_depth))
        # Left cushion
        pygame.draw.rect(screen, cushion_color, (70, 80, cushion_depth, HEIGHT-160))
        # Right cushion
        pygame.draw.rect(screen, cushion_color, (WIDTH-80, 80, cushion_depth, HEIGHT-160))
       
       
       
        # Draw pockets/holes with nice 3D effect
        for hx, hy in holes:
            # Draw pocket shadow
            pygame.draw.circle(screen, (0, 0, 0), (hx+2, hy+2), HOLE_RADIUS+2)
            # Draw pocket hole
            pygame.draw.circle(screen, (10, 10, 10), (hx, hy), HOLE_RADIUS)
            # Draw pocket rim
            pygame.draw.circle(screen, DARK_WOOD, (hx, hy), HOLE_RADIUS+5, 5)

    def draw_stylish_table(self, screen):
        self._draw_vertical_gradient(screen, (4, 14, 31), (7, 38, 60))

        shadow = pygame.Rect(42, 54, WIDTH - 84, HEIGHT - 94)
        pygame.draw.rect(screen, (0, 0, 0), shadow, border_radius=30)

        outer = pygame.Rect(50, 45, WIDTH - 100, HEIGHT - 105)
        rail = pygame.Rect(70, 65, WIDTH - 140, HEIGHT - 145)
        felt = pygame.Rect(92, 92, WIDTH - 184, HEIGHT - 184)

        pygame.draw.rect(screen, (38, 47, 58), outer, border_radius=28)
        pygame.draw.rect(screen, (205, 214, 215), outer, 5, border_radius=28)
        pygame.draw.rect(screen, (143, 60, 34), rail, border_radius=22)
        pygame.draw.rect(screen, (236, 126, 68), rail, 3, border_radius=22)
        pygame.draw.rect(screen, (26, 170, 226), felt, border_radius=12)
        pygame.draw.rect(screen, (117, 230, 255), felt, 2, border_radius=12)

        # Subtle felt stripes.
        stripe_w = max(24, felt.width // 18)
        for i, x in enumerate(range(felt.left, felt.right, stripe_w)):
            color = (24, 158, 214) if i % 2 == 0 else (31, 181, 235)
            pygame.draw.rect(screen, color, (x, felt.top, stripe_w, felt.height))
        pygame.draw.line(screen, (143, 230, 255), (felt.left + 120, felt.centery), (felt.right - 120, felt.centery), 2)
        pygame.draw.rect(screen, (117, 230, 255), felt, 2, border_radius=12)

        # Rail diamonds.
        diamond_color = (222, 202, 151)
        for x in range(felt.left + felt.width // 6, felt.right, felt.width // 6):
            pygame.draw.polygon(screen, diamond_color, [(x, 76), (x + 6, 82), (x, 88), (x - 6, 82)])
            pygame.draw.polygon(screen, diamond_color, [(x, HEIGHT - 82), (x + 6, HEIGHT - 76), (x, HEIGHT - 70), (x - 6, HEIGHT - 76)])
        for y in range(felt.top + felt.height // 4, felt.bottom, felt.height // 4):
            pygame.draw.polygon(screen, diamond_color, [(76, y), (82, y + 6), (88, y), (82, y - 6)])
            pygame.draw.polygon(screen, diamond_color, [(WIDTH - 88, y), (WIDTH - 82, y + 6), (WIDTH - 76, y), (WIDTH - 82, y - 6)])

        for hx, hy in holes:
            pygame.draw.circle(screen, (1, 5, 7), (hx + 4, hy + 5), HOLE_RADIUS + 8)
            pygame.draw.circle(screen, (3, 10, 12), (hx, hy), HOLE_RADIUS + 4)
            pygame.draw.circle(screen, (212, 219, 218), (hx, hy), HOLE_RADIUS + 8, 4)
   
    def draw_ball_counters(self, screen):
    # Draw counters for each player's remaining balls
        for i, player in enumerate(self.players):
            # Determine position based on player
            x_pos = 250 if i == 0 else WIDTH - 400
            y_pos = 30
           
            # Draw player info background
            pygame.draw.rect(screen, (40, 40, 40, 180), (x_pos, y_pos, 150, 80), 0, border_radius=10)
           
            # Highlight current player with thicker border
            border_thickness = 5 if i == self.current_player else 3
            pygame.draw.rect(screen, player["color"], (x_pos, y_pos, 150, 80), border_thickness, border_radius=10)
           
            # Draw player name with border
            name_text = player["name"]
            name_rendered = score_font.render(name_text, True, WHITE)
            name_width = name_rendered.get_width()
            text_x = x_pos + 75 - name_width//2
            text_y = y_pos + 10
           
            # Draw border by rendering multiple offset versions
            border_thickness = 2
            for off_x in range(-border_thickness, border_thickness+1):
                for off_y in range(-border_thickness, border_thickness+1):
                    if off_x != 0 or off_y != 0:  # Skip the center
                        border_render = score_font.render(name_text, True, BLACK)
                        screen.blit(border_render, (text_x + off_x, text_y + off_y))
           
            # Draw the main text on top
            screen.blit(name_rendered, (text_x, text_y))
           
            # Draw ball type (show ??? if not assigned yet)
            ball_type = player["type"]
            type_text = f"Type: {ball_type.capitalize()}" if ball_type else "Type: ???"
            type_rendered = game_font.render(type_text, True, WHITE)
            screen.blit(type_rendered, (x_pos + 75 - type_rendered.get_width()//2, y_pos + 35))
           
            # Draw remaining balls (show 7 if not assigned yet)
            remaining = self.count_remaining_balls(i)
            remain_text = game_font.render(f"Remaining: {remaining}", True, WHITE)
            screen.blit(remain_text, (x_pos + 75 - remain_text.get_width()//2, y_pos + 55))
 
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
        panel_w, panel_h = 190, 78
        positions = [(120, 28), (WIDTH - panel_w - 120, 28)]
        for i, player in enumerate(self.players):
            x, y = positions[i]
            selected = i == self.current_player
            panel = pygame.Rect(x, y, panel_w, panel_h)
            pygame.draw.rect(screen, (12, 23, 28), panel, border_radius=8)
            pygame.draw.rect(screen, player["color"] if selected else (125, 139, 136), panel, 3 if selected else 1, border_radius=8)

            name_text = score_font.render(player["name"], True, WHITE)
            screen.blit(name_text, (x + 14, y + 10))
            ball_type = player["type"].capitalize() if player["type"] else "Open"
            meta = game_font.render(f"{ball_type} | Remaining {self.count_remaining_balls(i)}", True, (218, 226, 222))
            screen.blit(meta, (x + 14, y + 44))
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
            pill = pygame.Rect(WIDTH // 2 - 170, 24, 340, 46)
            pygame.draw.rect(screen, (10, 20, 24), pill, border_radius=8)
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
