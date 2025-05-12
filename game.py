import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from DoubleDroneGym import DoubleDroneGym
import os
import time

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 400, 200
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = pygame.font.Font(None, 36)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Simulation")

def display_text(screen, text, x, y, color=BLACK):
    """Helper function to render text on the Pygame window."""
    text_surface = FONT.render(text, True, color)
    screen.blit(text_surface, (x, y))
import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from DoubleDroneGym import DoubleDroneGym
import os
import time

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 400, 200
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = pygame.font.Font(None, 36)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Simulation")

def display_text(screen, text, x, y, color=BLACK):
    """Helper function to render text on the Pygame window."""
    text_surface = FONT.render(text, True, color)
    screen.blit(text_surface, (x, y))

def wait_for_enter():
    """Show start screen and wait for the Enter key."""
    waiting = True
    while waiting:
        screen.fill(WHITE)
        display_text(screen, "Press ENTER to Start", 50, 80)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                waiting = False  # Exit loop when Enter is pressed


def game():
    env = DoubleDroneGym(game=True)
    model = PPO.load('models2/ppo500000.zip')
    obs = env.reset()[0]

    wait_for_enter()

    start_time = time.time()
    running = True

    while running:
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        env.render()

        # Calculate score (time since start)
        elapsed_time = int(time.time() - start_time)

        # Pygame event loop
        screen.fill(WHITE)
        display_text(screen, f"Time: {elapsed_time} s", 50, 50)

        if env.game_over():  # Collision threshold
            display_text(screen, "Game Over! Press r to restart", 50, 100)
            pygame.display.flip()
            
            waiting_for_restart = True
            while waiting_for_restart:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting_for_restart = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            # Restart environment
                            obs = env.reset()[0]
                            start_time = time.time()
                            waiting_for_restart = False
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                            waiting_for_restart = False
            continue  # Skip rest of loop to restart or quit

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if done:
            break

    env.close()
    pygame.quit()

if __name__ == "__main__":
    game()
