import pygame
import random
import math
import os

pygame.init()

WIDTH, HEIGHT = 700, 500
FPS = 60

BLUE = (78, 192, 202)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

CHARACTER_X = WIDTH // 2 - 200
CHARACTER_RADIUS = 20
CHARACTER_VEL = 2.75

WALL_WIDTH, WALL_HEIGHT = 50, HEIGHT
WALL_VEL = 3

GRAVITY = 0.125

current_path = os.path.dirname(os.path.realpath(__file__))

SCORE_FONT = pygame.font.Font(os.path.join(current_path, "resources/fb-font.ttf"), 80)
SPRITE = pygame.image.load(os.path.join(current_path, "resources/sprite.png"))
BOT_PIPE = pygame.image.load(os.path.join(current_path, "resources/bot_pipe.png"))
TOP_PIPE = pygame.image.load(os.path.join(current_path, "resources/top_pipe.png"))
BACKGROUND = pygame.image.load(os.path.join(current_path, "resources/background.png"))
FLOOR = pygame.image.load(os.path.join(current_path, "resources/floor.png"))


class Bird:
    """The player character with gravity-based physics."""

    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.y += random.uniform(-0.3, 0.3) * HEIGHT
        self.y_speed = 0
        self.radius = radius

    def draw(self, win):
        if self.y_speed != 0:
            angle = self.y_speed * 45 / (2 * CHARACTER_VEL)
            angle = max(angle, -80)
            bird_sprite = pygame.transform.rotate(SPRITE, angle)
        else:
            bird_sprite = SPRITE
        win.blit(bird_sprite, (self.x - self.radius, self.y - self.radius))

    def move(self, up=0):
        if up == 1:
            self.y_speed = CHARACTER_VEL
        if self.y + self.radius < HEIGHT:
            self.y -= self.y_speed
        self.y_speed -= GRAVITY

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y + random.uniform(-0.3, 0.3) * HEIGHT
        self.y_speed = 0


class Wall:
    """Obstacle pipe with dynamic hole sizing and progressive difficulty."""

    WALL_SPEED = WALL_VEL
    LAST_HOLE = HEIGHT // 2

    def __init__(self, x, y, is_second=False):
        self.x = self.original_x = x
        self.is_second = is_second
        if is_second:
            self.x = 3 * self.original_x // 2 + WALL_WIDTH // 2
        self.y = self.original_y = y
        self.height = WALL_HEIGHT
        self.width = WALL_WIDTH
        self.hole_size = CHARACTER_RADIUS * 8
        self.new_hole = -1
        self.hole = HEIGHT // 2 - int(random.uniform(-0.6, 0.6) * HEIGHT // 2)
        self.score = 0

    def draw(self, win):
        top_height = self.hole - self.hole_size // 2
        bot_height = self.hole + self.hole_size // 2
        win.blit(TOP_PIPE, (self.x - self.width // 2 - 1, -HEIGHT + top_height))
        win.blit(BOT_PIPE, (self.x - self.width // 2 - 1, bot_height))

    def move(self):
        self.x -= self.WALL_SPEED
        if self.x < CHARACTER_X - 3 * CHARACTER_RADIUS // 2 and self.new_hole < 0:
            self.compute_new_hole()
        if self.x < -self.width:
            self.reset()

    def compute_new_hole(self):
        """Generate next hole position, constrained by bird's movement capabilities."""
        self.new_hole = HEIGHT // 2 - int(random.uniform(-0.6, 0.6) * HEIGHT // 2)

        max_dist_top = ((WIDTH // 2) / Wall.WALL_SPEED) * CHARACTER_VEL * 0.7
        if Wall.LAST_HOLE - self.new_hole > max_dist_top:
            self.new_hole = Wall.LAST_HOLE - max_dist_top

        max_dist_bot = ((WIDTH // 2) / Wall.WALL_SPEED) * GRAVITY * FPS * 0.5
        if self.new_hole - Wall.LAST_HOLE > max_dist_bot:
            self.new_hole = Wall.LAST_HOLE + max_dist_bot

        Wall.LAST_HOLE = self.new_hole
        self.score += 1

    def reset(self):
        """Reset pipe position and increase difficulty based on score."""
        self.x = self.original_x
        self.y = self.original_y
        Wall.WALL_SPEED = WALL_VEL + WALL_VEL * self.score / 100
        self.hole = self.new_hole
        self.new_hole = -1
        hole_size = CHARACTER_RADIUS * 8 - self.score // 1.5
        self.hole_size = max(hole_size, CHARACTER_RADIUS * 1.05)

    def reset_original(self):
        Wall.WALL_SPEED = WALL_VEL
        self.__init__(self.original_x, self.original_y, self.is_second)


class FlappyBird:
    """Game engine with reward system for reinforcement learning agents."""

    def __init__(self, vis=True):
        self.vis = vis
        self.width = WIDTH
        self.height = HEIGHT
        self.frame_iteration = 0
        self.floor_x = 0

        if self.vis:
            self.window = pygame.display.set_mode((WIDTH, HEIGHT + 50))
            pygame.display.set_caption("Flappy Bird")
            self.clock = pygame.time.Clock()

        self.character = Bird(CHARACTER_X, HEIGHT // 2, CHARACTER_RADIUS)
        self.walls = [Wall(WIDTH, 0), Wall(WIDTH, 0, True)]
        self.score = 0
        self.current_wall = self.walls[0]

    def draw(self):
        self.window.fill(BLUE)
        self.window.blit(BACKGROUND, (0, 0))
        self.character.draw(self.window)

        for wall in self.walls:
            wall.draw(self.window)

        score_text = SCORE_FONT.render(f"{self.score}", 1, WHITE)
        shadow_text = SCORE_FONT.render(f"{self.score}", 1, BLACK)
        self.window.blit(shadow_text, (WIDTH // 2 - score_text.get_width() // 2 + 5, 25))
        self.window.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))
        self.window.blit(FLOOR, (self.floor_x, HEIGHT))

        pygame.display.update()

    def move(self, action):
        self.character.move(action)
        for wall in self.walls:
            wall.move()
        self.floor_x -= Wall.WALL_SPEED
        if self.floor_x < -432:
            self.floor_x = 0

    def handle_collision(self):
        """Check for collisions with boundaries and pipes (including pipe corners)."""
        if self.character.y + self.character.radius >= HEIGHT or self.character.y < 0:
            return True

        for wall in self.walls:
            if wall.x - WALL_WIDTH // 2 >= CHARACTER_X + CHARACTER_RADIUS:
                continue
            if wall.x + WALL_WIDTH // 2 <= CHARACTER_X - CHARACTER_RADIUS:
                continue

            # Bird is horizontally inside the pipe
            if wall.x - WALL_WIDTH // 2 <= CHARACTER_X <= wall.x + WALL_WIDTH // 2:
                if (self.character.y - CHARACTER_RADIUS < wall.hole - wall.hole_size // 2
                        or self.character.y + CHARACTER_RADIUS > wall.hole + wall.hole_size // 2):
                    return True
                continue

            # Check corner collisions using distance
            if wall.x + WALL_WIDTH // 2 < CHARACTER_X:
                delta_x = CHARACTER_X - (wall.x + WALL_WIDTH // 2)
            else:
                delta_x = wall.x - WALL_WIDTH // 2 - CHARACTER_X

            dist_y_top = math.sqrt(delta_x ** 2 + (wall.hole - wall.hole_size // 2 - self.character.y) ** 2)
            dist_y_bot = math.sqrt(delta_x ** 2 + (wall.hole + wall.hole_size // 2 - self.character.y) ** 2)
            if dist_y_top < CHARACTER_RADIUS or dist_y_bot < CHARACTER_RADIUS:
                return True

        return False

    def compute_score(self):
        return sum(wall.score for wall in self.walls)

    def reset(self):
        self.frame_iteration = 0
        self.score = 0
        self.character.reset()
        for wall in self.walls:
            wall.reset_original()
        self.current_wall = self.walls[0]
        if self.vis:
            self.draw()

    def play_step(self, action):
        """Execute one game step and return (reward, game_over, score).

        Reward policy:
            +50  passing through a pipe
            +1   moving toward the pipe gap
            -1   moving away from the pipe gap
            -1   collision (game over)
        """
        if self.vis:
            self.draw()
            self.clock.tick(FPS)

        game_over = False
        self.frame_iteration += 1

        if self.vis:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        self.score = self.compute_score()
        past_score = self.score

        self.move(action)

        if self.walls[1].score < self.walls[0].score:
            self.current_wall = self.walls[1]
        else:
            self.current_wall = self.walls[0]

        if self.handle_collision():
            game_over = True
            self.reset()

        # Reward: encourage moving toward the pipe gap
        if self.current_wall.hole >= self.character.y:
            reward = 1 if action == 0 else -1
        else:
            reward = 1 if action == 1 else -1

        # Bonus for passing a pipe
        if self.compute_score() > past_score:
            reward = 50

        if game_over:
            reward = -1

        return reward, game_over, past_score
