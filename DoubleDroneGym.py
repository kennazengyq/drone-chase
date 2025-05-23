import numpy as np 
import cv2
import matplotlib.pyplot as plt
import gymnasium as gym
import random
from Multiple_Quad import Multiple_Quad
from gymnasium import Env, spaces
from math import cos, sin
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3 import PPO

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

import pygame

pygame.init()

class DoubleDroneGym(Env):
    def __init__(self, game=False):
        super(DoubleDroneGym,self).__init__()
        self.game = game

        self.logger = configure("./logs", ["stdout", "csv"])


        self.a_x = 5
        self.a_y = 5
        self.a_z = 0

        self.b_x = random.uniform(0,10)
        self.b_y = random.uniform(0,10)
        self.b_z = random.uniform(0,10)

        self.x_vel = 0
        self.y_vel = 0
        self.z_vel = 0
        self.x_acc = 0
        self.y_acc = 0
        self.z_acc = 0

        self.m = 2
        self.g = 9.81

        self.dt = 0.05
        self.t = 0

        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.thrust = self.m*self.g

        self.target_x = 5
        self.target_y = 5
        self.target_z = 5

        self.quad = Multiple_Quad(a_x=self.a_x, a_y=self.a_y, a_z=self.a_z, b_x=self.b_x, b_y=self.b_y, b_z=self.b_z,roll=self.roll, pitch = self.pitch, yaw=self.yaw,show_animation=True)

        # thrust range: [0.8,1.5], yaw/pitch/roll range: [-1,1]
        self.action_space = spaces.Box(low=np.array([0.8, -1, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
        self.state_reward = self.calc_reward()

        self.draw_elements_on_state()
        self.b_mode = random.choice([1, 2, 3])  # 1: static, 2: evasive, 3: circular
        # self.b_mode = 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.state),), dtype=np.float32)
        self.count = 0
        self.episode_count = 0

    def calc_reward(self):
        # Relative vector and distance
        rel_x = self.a_x - self.b_x
        rel_y = self.a_y - self.b_y
        rel_z = self.a_z - self.b_z
        distance = np.linalg.norm([rel_x, rel_y, rel_z])

        # Reward for reducing distance
        distance_reward = -distance * 10 # Penalize distance

        # Strong reward for collision
        collision_reward = 0
        if distance < 0.5:  # Collision threshold
            collision_reward = 1000  # Large reward for collision
        # Penalize hovering near the target without colliding
        hover_penalty = 0
        if 0.3 < distance < 2:  # If hovering close but not colliding
            hover_penalty = -10

        # Total reward
        total_reward = distance_reward + collision_reward + hover_penalty
        return total_reward

    # def calc_reward(self):
    #     # Relative vector and distance
    #     rel_x = self.a_x - self.b_x
    #     rel_y = self.a_y - self.b_y
    #     rel_z = self.a_z - self.b_z
    #     rel_vector = np.array([rel_x, rel_y, rel_z])
    #     distance = np.linalg.norm(rel_vector)

    #     # Velocity alignment with chase direction
    #     velocity_vector = np.array([self.x_vel, self.y_vel, self.z_vel])
    #     if np.linalg.norm(velocity_vector) > 0:
    #         approach_bonus = np.dot(velocity_vector, -rel_vector) / (np.linalg.norm(velocity_vector) * distance + 1e-6)
    #     else:
    #         approach_bonus = 0

    #     # Proximity reward
    #     proximity_reward = np.exp(-distance * 2.0) * 20  # Increase weight for proximity
    #     if distance < 0.1:
    #         proximity_reward += 100

    #     # Catching reward for minimizing time to catch
    #     catching_reward = 0
    #     if distance < 0.2:
    #         catching_reward = 500 - self.t * 10  # Reward inversely proportional to time taken

    #     # Total reward
    #     total_reward = (
    #         proximity_reward +
    #         approach_bonus +
    #         catching_reward
    #     )

    #     return total_reward



    
    def reset(self, seed=0):
        super().reset(seed=seed)
        self.ep_return = 0

        self.t = 0

        self.a_x = 5
        self.a_y = 5
        self.a_z = 0

        self.b_x = random.uniform(0,10)
        self.b_y = random.uniform(0,10)
        self.b_z = random.uniform(0,10)

        self.x_vel = 0
        self.y_vel = 0
        self.z_vel = 0
        self.x_acc = 0
        self.y_acc = 0
        self.z_acc = 0

        self.m = 2
        self.g = 9.81
        self.torque_step = 0.001
        self.thrust_step = 0.02

        self.dt = 0.05
        self.t = 0

        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.thrust = self.m*self.g

        self.target_x = 5
        self.target_y = 5
        self.target_z = 5

        self.quad.update_pose(a_x=self.a_x, a_y=self.a_y, a_z=self.a_z, b_x=self.b_x, b_y=self.b_y, b_z=self.b_z, roll=self.roll, pitch=self.pitch, yaw=self.yaw)

        self.state_reward = self.calc_reward()

        self.draw_elements_on_state()
        self.b_mode = random.choice([1, 2, 3])  # 1: static, 2: evasive, 3: circular
        # self.b_mode = 2

        self.episode_count += 1
        # print("mode", self.b_mode)

        
        return self.state, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            pass
            
        elif mode == 'rgb_array':
            return self.state
        
    def close(self):
        cv2.destroyAllWindows()

    def draw_elements_on_state(self):
        # Compute relative position
        rel_x = self.a_x - self.b_x
        rel_y = self.a_y - self.b_y
        rel_z = self.a_z - self.b_z

        self.state = np.array([
            rel_x, rel_y, rel_z,  # Relative position
            self.x_vel, self.y_vel, self.z_vel,  # Velocity
            self.roll, self.pitch, self.yaw, self.thrust  # Orientation and thrust
        ], dtype=np.float32)


    def step(self, action):
        done = False
        assert self.action_space.contains(action), "invalid action"

        self.b_step(game=self.game)

        thrust_factor = action[0]
        self.thrust = np.clip(thrust_factor * self.m * self.g, 0.8 * self.m * self.g, 3 * self.m * self.g)
        
        self.roll += action[1] * 0.05  
        self.pitch += action[2] * 0.05  
        self.yaw += action[3] * 0.05  

        self.roll = np.clip(self.roll, -0.5, 0.5)  
        self.pitch = np.clip(self.pitch, -0.5, 0.5)  
        self.yaw = np.clip(self.yaw, -np.pi, np.pi)  

        R = self.rotation_matrix(self.roll, self.pitch, self.yaw)
        thrust_array = np.array(self.thrust)
        acc = (np.matmul(R, np.array(
            [0, 0, thrust_array.item()]).T) - np.array([0, 0, self.m * self.g]).T) / self.m

        # Calculate distance between Drone A and Drone B
        distance = np.linalg.norm([self.a_x - self.b_x, self.a_y - self.b_y, self.a_z - self.b_z])
        # Scale acceleration based on proximity to Drone B
        # scaling_factor = 0.5 * max(1.0, 10 / (distance + 1e-6))  # Increase acceleration as distance decreases
        scaling_factor = 1
        self.x_acc = acc[0] * scaling_factor
        self.y_acc = acc[1] * scaling_factor
        self.z_acc = acc[2] * scaling_factor

        self.x_vel += self.x_acc * self.dt
        self.y_vel += self.y_acc * self.dt
        self.z_vel += self.z_acc * self.dt
        self.a_x += self.x_vel * self.dt
        self.a_y += self.y_vel * self.dt
        self.a_z += self.z_vel * self.dt

        self.a_x, self.x_vel, enforce_x = self.enforce_boundary(self.a_x, self.x_vel, 0, 10)
        self.a_y, self.y_vel, enforce_y = self.enforce_boundary(self.a_y, self.y_vel, 0, 10)
        self.a_z, self.z_vel, enforce_z = self.enforce_boundary(self.a_z, self.z_vel, 0, 10)
        
        if enforce_x or enforce_y:
            self.roll, self.pitch, self.yaw = 0, 0, 0
        
        if enforce_z:
            if self.a_z == 0:
                self.thrust = self.m * self.g * 1.2
            else:
                self.thrust = self.m * self.g * 0.8

        self.quad.update_pose(a_x=self.a_x, a_y=self.a_y, a_z=self.a_z, b_x=self.b_x, b_y=self.b_y, b_z=self.b_z, roll=self.roll, pitch=self.pitch, yaw=self.yaw)
        
        self.t += self.dt
        self.draw_elements_on_state()
        self.state_reward = self.calc_reward()
        if hasattr(self, "logger"):
            self.logger.record("reward", self.state_reward)
        self.count += 1
        if self.count % 100 == 0:
            print(distance, self.state_reward)
        # Check for collision
        if self.game:
            if distance < 0.3:
                print("disance < 0.3:", distance, self.state_reward)
                done = True
            else:
                done = False
        else:
            if distance < 0.1:
                print("disance < 0.1:", distance, self.state_reward)
                done = True
            else:
                done = False

        return self.state, self.state_reward, done, done, {}


    def b_step(self, game=False):
        if game:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:  
                self.b_y = min(10, self.b_y + 0.15)
            elif keys[pygame.K_x]:
                self.b_y = max(0, self.b_y - 0.15)
            
            if keys[pygame.K_RIGHT]:
                self.b_x = min(10, self.b_x + 0.15)
            elif keys[pygame.K_LEFT]:
                self.b_x = max(0, self.b_x - 0.15)

            if keys[pygame.K_UP]:
                self.b_z = min(10, self.b_z + 0.15)
            elif keys[pygame.K_DOWN]:
                self.b_z = max(0, self.b_z - 0.15)

        else:
            if self.b_mode == 1:
                # Static - blue drone does not move
                pass

            elif self.b_mode == 2:
                # Evasive - move away from red drone
                direction = np.array([self.b_x - self.a_x, self.b_y - self.a_y, self.b_z - self.a_z])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                    speed = 0.2
                    self.b_x += direction[0] * speed
                    self.b_y += direction[1] * speed
                    self.b_z += direction[2] * speed

                # Check boundaries and reverse direction if stuck
                if self.b_x <= 0 or self.b_x >= 10:
                    self.b_x -= direction[0] * speed
                if self.b_y <= 0 or self.b_y >= 10:
                    self.b_y -= direction[1] * speed
                if self.b_z <= 0 or self.b_z >= 10:
                    self.b_z -= direction[2] * speed

            elif self.b_mode == 3:
                # Circular motion around fixed center point with noise and vertical oscillation
                if not hasattr(self, "circle_angle"):
                    self.circle_angle = 0
                    self.circle_center = np.array([random.randint(3,6), random.randint(3,6)])
                    self.circle_radius = random.randint(2,5)
                    self.circle_speed = 0.08  # radians per step
                    self.vertical_phase = 0  # Phase for vertical oscillation
                    self.vertical_amplitude = random.gauss(1,1)  # Amplitude for vertical oscillation
                    self.vertical_speed = 0.05  # Speed of vertical oscillation

                # Add noise to the center and radius
                noise_center = np.random.normal(0, 0.1, size=2)  # Small noise for center
                noise_radius = np.random.normal(0, 0.05)  # Small noise for radius
                noisy_center = self.circle_center + noise_center
                noisy_radius = self.circle_radius + noise_radius

                self.circle_angle += self.circle_speed
                self.b_x = noisy_center[0] + noisy_radius * np.cos(self.circle_angle)
                self.b_y = noisy_center[1] + noisy_radius * np.sin(self.circle_angle)

                # Vertical oscillation
                self.vertical_phase += 0.1  # Adjust speed of vertical oscillation
                self.b_z = 5 + 0.5 * np.sin(self.vertical_phase) + np.random.normal(0, 0.05)  # Oscillate around altitude 5 with noise


        # Clip to bounds
        self.b_x = np.clip(self.b_x, 0, 10)
        self.b_y = np.clip(self.b_y, 0, 10)
        self.b_z = np.clip(self.b_z, 0, 10)

    def game_over(self):
        rel_x = self.a_x - self.b_x
        rel_y = self.a_y - self.b_y
        rel_z = self.a_z - self.b_z
        distance = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)

        if self.game:
            return distance < 0.3
        else:
            return distance < 0.1
        

    def rotation_matrix(self, roll, pitch, yaw):
        """
        Calculates the ZYX rotation matrix.

        Args
            Roll: Angular position about the x-axis in radians.
            Pitch: Angular position about the y-axis in radians.
            Yaw: Angular position about the z-axis in radians.

        Returns
            3x3 rotation matrix as NumPy array
        """
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll)],
            [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) *
            sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll)],
            [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw)]
            ])

    def enforce_boundary(self, position, velocity, min_bound, max_bound):
        """
        Constrains a position within a given boundary and adjusts velocity.

        Args:
            position: The current position value (x, y, or z).
            velocity: The current velocity value (x, y, or z).
            min_bound: Minimum allowed value for the position.
            max_bound: Maximum allowed value for the position.

        Returns:
            Tuple of (clamped_position, adjusted_velocity).
        """
        if position < min_bound:
            return min_bound, 0, True # Clamp position to minimum and reset velocity
        elif position > max_bound:
            return max_bound, 0, True  # Clamp position to maximum and reset velocity
        return position, velocity, False
    

# env = DoubleDroneGym(game=False)
# obs = env.reset()[0]
# model = PPO.load('models/ppo800000.zip')
# while True:
#     # Take a random action
#     currStep = model.predict(observation=obs, deterministic=True)[0]
#     obs, reward, done, _, _ = env.step(currStep)


#     # Render the game
#     env.render('human')

#     if done == True:
#         break


# env.close()