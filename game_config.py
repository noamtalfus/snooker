import pygame
import random

pygame.init()
pygame.font.init()
WIDTH, HEIGHT = 1200, 800
TABLE_COLOR = (0, 128, 0)
WOOD_COLOR = (120, 60, 20)  
DARK_WOOD = (80, 40, 20)    
BALL_RADIUS = 14
MAX_POWER = 35
HOLE_RADIUS = 35         
WHITE = (255, 255, 255)
RED = (200, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (200, 200, 200)
BROWN = (150, 75, 0)
PINK = (255, 105, 180)


PLAYER1_COLOR = (200, 30, 30)
PLAYER2_COLOR = (30, 150, 30)  



screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
pygame.display.set_caption("8-Ball Pool")
clock = pygame.time.Clock()

title_font = pygame.font.SysFont('Arial', 48, bold=True)
menu_font = pygame.font.SysFont('Arial', 32)
score_font = pygame.font.SysFont('Arial', 24)
game_font = pygame.font.SysFont('Arial', 18)

def create_wood_texture(width, height, base_color, grain_color, grain_density=15):
    texture = pygame.Surface((width, height))
    texture.fill(base_color)
   
    # Add wood grain
    for i in range(0, width, grain_density):
        grain_width = random.randint(1, 3)
        grain_shade = random.randint(-20, 20)
        adjusted_color = (
            max(0, min(255, base_color[0] + grain_shade)),
            max(0, min(255, base_color[1] + grain_shade)),
            max(0, min(255, base_color[2] + grain_shade))
        )
        pygame.draw.line(texture, adjusted_color, (i, 0), (i, height), grain_width)
   
    # Add some knots
    for _ in range(width // 200):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        knot_radius = random.randint(5, 15)
        pygame.draw.circle(texture, grain_color, (x, y), knot_radius)
        pygame.draw.circle(texture, base_color, (x, y), knot_radius-2)
   
    return texture

light_wood_texture = create_wood_texture(WIDTH, HEIGHT, WOOD_COLOR, DARK_WOOD)
dark_wood_texture = create_wood_texture(WIDTH, HEIGHT, DARK_WOOD, (60, 30, 10))

def load_image(name, scale=1.0):
    surf = pygame.Surface((int(100 * scale), int(100 * scale)))
    surf.fill(WOOD_COLOR)
    pygame.draw.rect(surf, LIGHT_GRAY, (10, 10, 80, 80))
    pygame.draw.rect(surf, BLACK, (10, 10, 80, 80), 2)
    return surf


def play_sound(name):
    # Placeholder for sound effects
    pass

MENU = 0
GAME = 1
PAUSED = 2
GAME_OVER = 3
INSTRUCTIONS = 4
TRAINING = 5
TRAINING_MENU = 6


holes = [
    (80, 80),           
    (WIDTH//2, 70),     
    (WIDTH-80, 80),     
    (80, HEIGHT-80),    
    (WIDTH//2, HEIGHT-70), 
    (WIDTH-80, HEIGHT-80)  
]
