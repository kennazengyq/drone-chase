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
    env = DoubleDroneGym()
    model = PPO.load('ppo660000.zip')
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

        # Check for collision
        

        # Pygame event loop
        screen.fill(WHITE)
        display_text(screen, f"Time: {elapsed_time} s", 50, 50)

        if env.game_over():  # Collision threshold
            display_text(screen, "Game Over!", 50, 100)
            pygame.display.flip()
            time.sleep(2)  # Show game over message for 2 seconds
            break  # Exit the loop

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
