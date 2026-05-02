import math

import pygame

from game_config import *

class Ball:
    def __init__(self, x, y, color, number=0, radius=BALL_RADIUS, is_striped=False):
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
       
        # Check for table border collision
        cushion_dampening = 0.8  # Energy loss on cushion hit
       
        if self.x - self.radius < 80:
            self.x = 80 + self.radius
            self.vx = -self.vx * cushion_dampening
            play_sound("cushion")
        elif self.x + self.radius > WIDTH - 80:
            self.x = WIDTH - 80 - self.radius
            self.vx = -self.vx * cushion_dampening
            play_sound("cushion")
           
        if self.y - self.radius < 80:
            self.y = 80 + self.radius
            self.vy = -self.vy * cushion_dampening
            play_sound("cushion")
        elif self.y + self.radius > HEIGHT - 80:
            self.y = HEIGHT - 80 - self.radius
            self.vy = -self.vy * cushion_dampening
            play_sound("cushion")
   
    def draw(self, screen):
        # Ball base
        ball_surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(ball_surface, self.color, (self.radius, self.radius), self.radius)

        # Radial highlight (top-left for gloss)
        gloss_radius = int(self.radius * 0.5)
        gloss_surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(gloss_surface, (255, 255, 255, 60), (int(self.radius * 0.6), int(self.radius * 0.6)), gloss_radius)
        ball_surface.blit(gloss_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        # Shadow under ball
        shadow_offset = 3
        pygame.draw.circle(screen, (30, 30, 30), (int(self.x + shadow_offset), int(self.y + shadow_offset)), self.radius)

        # Blit ball
        screen.blit(ball_surface, (int(self.x - self.radius), int(self.y - self.radius)))

        # Draw number
        if self.number > 0:
            number_font = pygame.font.SysFont('Arial', 10, bold=True)
            text_color = WHITE if self.color == BLACK or self.color == BLUE else BLACK
            number_text = number_font.render(str(self.number), True, text_color)
            text_rect = number_text.get_rect(center=(int(self.x), int(self.y)))
            screen.blit(number_text, text_rect)

   
    def strike(self, angle, power):
        self.vx = power * math.cos(angle)
        self.vy = power * math.sin(angle)
        play_sound("cue_hit")
   
    def in_hole(self):
        for hx, hy in holes:
            if math.hypot(self.x - hx, self.y - hy) < HOLE_RADIUS:
                play_sound("ball_potted")
                self.potted = True
                return True
        return False
   
    def reset(self):
        self.x, self.y = self.original_pos
        self.vx, self.vy = 0, 0
        self.potted = False


class Cue:
    def __init__(self):
        self.angle = 0
        self.power = 0
        self.max_power = MAX_POWER
        self.pulling_back = False
        self.color = (200, 200, 150)
        self.power_accumulation_rate = 0.5  
        self.current_pull_distance = 0
        self.length = BALL_RADIUS * 11     
        self.width = BALL_RADIUS // 2     
        self.max_pull_distance = BALL_RADIUS * 5
        self.aim_line_length = BALL_RADIUS * 20

    def draw_predictive_guide(self, screen, cue_ball, other_balls):
        direction_x = math.cos(self.angle)
        direction_y = math.sin(self.angle)

        # Shorter guide for higher difficulty: no collision or bounce prediction
        max_guide_length = BALL_RADIUS * 8
        step_distance = 15
        steps = int(max_guide_length // step_distance)

        start_x = cue_ball.x
        start_y = cue_ball.y
        for step in range(1, steps + 1):
            dot_x = start_x + direction_x * step * step_distance
            dot_y = start_y + direction_y * step * step_distance
            pygame.draw.circle(screen, (255, 255, 255), (int(dot_x), int(dot_y)), 3)

    def draw(self, screen, ball_x, ball_y, all_balls=None):
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Always update angle to point from ball to mouse
        dx = mouse_x - ball_x
        dy = mouse_y - ball_y
        self.angle = math.atan2(dy, dx)

        # Direction vector
        direction_x = math.cos(self.angle)
        direction_y = math.sin(self.angle)

        # Pull distance if pulling
        if self.pulling_back:
            drag_distance = math.hypot(ball_x - mouse_x, ball_y - mouse_y)
            pull_distance = min(drag_distance, self.max_pull_distance)
        else:
            pull_distance = 20  # fixed offset when idle

        # Cue dimensions
        cue_length = self.length
        cue_width = self.width

        # Cue positions
        start_x = ball_x - direction_x * pull_distance
        start_y = ball_y - direction_y * pull_distance
        end_x = start_x - direction_x * cue_length
        end_y = start_y - direction_y * cue_length

        # Draw shadow
        shadow_offset = 3
        pygame.draw.line(screen, (50, 50, 50),
                         (start_x + shadow_offset, start_y + shadow_offset),
                         (end_x + shadow_offset, end_y + shadow_offset),
                         cue_width + 4)

        # Draw main cue
        pygame.draw.line(screen, self.color, (start_x, start_y), (end_x, end_y), cue_width)

        # Draw tip
        tip_length = cue_length * 0.1
        tip_x = start_x + direction_x * tip_length
        tip_y = start_y + direction_y * tip_length
        pygame.draw.line(screen, (160, 120, 40), (start_x, start_y), (tip_x, tip_y), cue_width)

        if all_balls:
            dummy_ball = Ball(ball_x, ball_y, WHITE, radius=BALL_RADIUS)
            self.draw_predictive_guide(screen, dummy_ball, all_balls)
