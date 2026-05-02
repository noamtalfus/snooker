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
        self.reset_game()
        self.state = MENU
        self.menu_options = ["New Game", "Play vs Trained AI", "Training Mode", "Instructions", "Quit"]
        self.selected_option = 0
        self.training = None
        self.training_selected_option = 0
        self.training_ball_count = 15
        self.training_random_balls = False
        self.training_opponent = "random"  # random | self | none
       
    def reset_game(self, ball_count=15, vs_trained_agent=False):
        self.active_ball_count = max(1, min(15, int(ball_count)))
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
       
        
        ball_numbers = list(range(1, 16))
        random.shuffle(ball_numbers)
       
        
        eight_idx = ball_numbers.index(8)
        other_idx = 4
        ball_numbers[eight_idx], ball_numbers[other_idx] = ball_numbers[other_idx], ball_numbers[eight_idx]
       
        selected_positions = ball_positions[:self.active_ball_count]
        selected_numbers = ball_numbers[:self.active_ball_count]

        for i, (offset_x, offset_y) in enumerate(selected_positions):
            number = selected_numbers[i]
            x = rack_start_x + offset_x * ball_diameter
            y = rack_start_y + offset_y * ball_diameter
            is_striped = number > 8
            ball_number = number if number != 8 else 8
            balls.append(Ball(x, y, ball_colors[number-1], ball_number, BALL_RADIUS, is_striped))
       
        return balls
   
    def handle_menu_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_option = (self.selected_option - 1) % len(self.menu_options)
            elif event.key == pygame.K_DOWN:
                self.selected_option = (self.selected_option + 1) % len(self.menu_options)
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

    def start_training(self, ball_count, random_balls, opponent_type):
        self.close_training()
        self.training = TrainingController(
            ball_count=ball_count,
            random_balls=random_balls,
            opponent_type=opponent_type,
            resume_checkpoint=False
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
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.training_selected_option = (self.training_selected_option - 1) % 4
            elif event.key == pygame.K_DOWN:
                self.training_selected_option = (self.training_selected_option + 1) % 4
            elif event.key == pygame.K_ESCAPE:
                self.state = MENU
            elif event.key == pygame.K_LEFT:
                if self.training_selected_option == 0:
                    self.training_ball_count = max(1, self.training_ball_count - 1)
                elif self.training_selected_option == 1:
                    self.training_random_balls = not self.training_random_balls
                elif self.training_selected_option == 2:
                    self.training_opponent = self._cycle_opponent(-1)
            elif event.key == pygame.K_RIGHT:
                if self.training_selected_option == 0:
                    self.training_ball_count = min(15, self.training_ball_count + 1)
                elif self.training_selected_option == 1:
                    self.training_random_balls = not self.training_random_balls
                elif self.training_selected_option == 2:
                    self.training_opponent = self._cycle_opponent(1)
            elif event.key == pygame.K_RETURN:
                if self.training_selected_option == 3:
                    self.start_training(self.training_ball_count, self.training_random_balls, self.training_opponent)
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
   
    def draw_menu(self, screen):
        screen.blit(dark_wood_texture, (0, 0))
        title_shadow = title_font.render("8-Ball Pool", True, BLACK)
        title_text = title_font.render("8-Ball Pool", True, WHITE)
        screen.blit(title_shadow, (WIDTH//2 - title_shadow.get_width()//2 + 2, HEIGHT//4 + 2))
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, HEIGHT//4))
       
        for i, option in enumerate(self.menu_options):
            color = WHITE if i == self.selected_option else LIGHT_GRAY
            text = menu_font.render(option, True, color)
            text_rect = text.get_rect(center=(WIDTH//2, HEIGHT//2 + i * 50))
           
            if i == self.selected_option:
                pygame.draw.rect(screen, PLAYER1_COLOR,
                               (text_rect.left - 20, text_rect.top - 5,
                                text_rect.width + 40, text_rect.height + 10),
                               3, border_radius=10)
           
            screen.blit(text, text_rect)

        if self.ai_status_text:
            status_color = RED if "failed" in self.ai_status_text.lower() or "missing" in self.ai_status_text.lower() else LIGHT_GRAY
            status_text = game_font.render(self.ai_status_text, True, status_color)
            screen.blit(status_text, (WIDTH//2 - status_text.get_width()//2, HEIGHT - 70))
   
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

    def _cycle_opponent(self, direction):
        order = ["random", "self", "none"]
        idx = order.index(self.training_opponent)
        idx = (idx + direction) % len(order)
        return order[idx]

    def draw_training_menu(self, screen):
        screen.blit(dark_wood_texture, (0, 0))
        title_text = title_font.render("Training Settings", True, WHITE)
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, HEIGHT//4))

        options = [
            f"Ball Count: {self.training_ball_count}",
            f"Random Ball Positions: {'On' if self.training_random_balls else 'Off'}",
            f"Opponent: {self._opponent_label(self.training_opponent)}",
            "Start Training"
        ]

        for i, label in enumerate(options):
            color = WHITE if i == self.training_selected_option else LIGHT_GRAY
            text = menu_font.render(label, True, color)
            text_rect = text.get_rect(center=(WIDTH//2, HEIGHT//2 + i * 50))

            if i == self.training_selected_option:
                pygame.draw.rect(screen, PLAYER1_COLOR,
                               (text_rect.left - 20, text_rect.top - 5,
                                text_rect.width + 40, text_rect.height + 10),
                               3, border_radius=10)

            screen.blit(text, text_rect)

        help_text = game_font.render("Use LEFT/RIGHT to change values, ENTER to start", True, LIGHT_GRAY)
        screen.blit(help_text, (WIDTH//2 - help_text.get_width()//2, HEIGHT - 110))
        back_text = menu_font.render("Press ESC to return to menu", True, LIGHT_GRAY)
        screen.blit(back_text, (WIDTH//2 - back_text.get_width()//2, HEIGHT - 70))

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

        overlay = pygame.Surface((WIDTH, 110), pygame.SRCALPHA)
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
       








    def draw_game(self, screen):
        # Draw table and elements
        self.draw_table(screen)
       
        # Draw balls
        for ball in self.balls:
            if not ball.potted:
                ball.draw(screen)
       
        if not self.cue_ball.potted:
            self.cue_ball.draw(screen)
       
        # Draw cue when a human player is aiming
        if not self.balls_moving and not self.cue_ball.potted and not self.is_ai_turn():
            self.cue.draw(screen, self.cue_ball.x, self.cue_ball.y, self.balls)
       
        # Draw ball counters
        self.draw_ball_counters(screen)
       
        # Draw current player indicator
        current_player = self.players[self.current_player]
        indicator_text = score_font.render(f"Current Turn: {current_player['name']}", True, current_player["color"])
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
