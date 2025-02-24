import numpy as np 
import cv2
import matplotlib.pyplot as plt
import gymnasium as gym
import random
from Multiple_Quad import Multiple_Quad
from gymnasium import Env, spaces
from math import cos, sin
from stable_baselines3.common.logger import Logger, configure

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

import pygame

pygame.init()

class DoubleDroneGym(Env):
    def __init__(self):
        super(DoubleDroneGym,self).__init__()

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

        # thrust range: [0.8, 2], yaw/pitch/roll range: [-1,1]
        self.action_space = spaces.Box(low=np.array([0.8, -1, -1, -1]), high=np.array([1.5, 1, 1, 1]), dtype=np.float32)

        self.state_reward = self.calc_reward()

        self.draw_elements_on_state()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.state),), dtype=np.float32)
        self.count = 0


    def calc_reward(self):
        # Relative Position
        rel_x = self.a_x - self.b_x
        rel_y = self.a_y - self.b_y
        rel_z = self.a_z - self.b_z
        distance = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)

        velocity_vector = np.array([self.x_vel, self.y_vel, self.z_vel])
        
        # Distance Reward (negative reward for being far)
        distance_penalty = -distance * 0.5  

        # Velocity Control Penalty (avoid high speeds)
        velocity_penalty = -0.1 * np.linalg.norm(velocity_vector)

        # Bonus for being close to B
        if distance < 0.5:
            close_bonus = 10  
        else:
            close_bonus = 0

        if self.a_z == 0 or self.a_z == 10:
            distance_penalty += -10
        if self.a_y == 0 or self.a_y == 10:
            distance_penalty += -5
        if self.a_x == 0 or self.a_x == 10:
            distance_penalty += -5

        rel_pos = np.array([rel_x, rel_y, rel_z])
        

        approach_reward = np.dot(rel_pos, velocity_vector) / (np.linalg.norm(rel_pos) + 1e-6)  
        reward = distance_penalty + velocity_penalty + close_bonus + approach_reward


        return reward


    
    def reset(self, seed=0):
        super().reset(seed=seed)
        self.ep_return = 0

        self.t = 0

        self.a_x = 5
        self.a_y = 5
        self.a_z = 0

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

        self.b_step()


        thrust_factor = action[0]
        self.thrust = np.clip(thrust_factor * self.m * self.g, 0.8 * self.m * self.g, 2.5 * self.m * self.g)
        
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
        
        self.x_acc = acc[0]
        self.y_acc = acc[1]
        self.z_acc = acc[2]
        self.x_vel += self.x_acc * self.dt
        self.y_vel += self.y_acc * self.dt
        self.z_vel += self.z_acc * self.dt
        self.a_x += self.x_vel * self.dt
        self.a_y += self.y_vel * self.dt
        self.a_z += self.z_vel * self.dt
        self.a_z, self.z_vel, enforce_z = self.enforce_boundary(self.a_z, self.z_vel, 0, 10)

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
        if self.count % 20 == 0:
            print(action, self.state_reward)
        self.count += 1

        # if np.linalg.norm([self.a_x - self.b_x, self.a_y - self.b_y, self.a_z - self.b_z]) < 0.2:
        #     done = True

        return self.state, self.state_reward, done, False, {}


    def b_step(self):
        # # Make B randomly move
        # amplitude = 2  # Controls how far B moves
        # frequency = 0.5  # Adjust frequency for speed of oscillation

        # noise = np.random.uniform(-0.05, 0.05, 3)  # Small random noise
        # self.b_x = np.clip(5 + amplitude * np.sin(frequency * self.t) + noise[0], 0, 10)
        # self.b_y = np.clip(5 + amplitude * np.cos(frequency * self.t) + noise[1], 0, 10)
        # self.b_z = np.clip(5 + (amplitude / 2) * np.sin(frequency * self.t / 2) + noise[2], 0, 10)

        # # Keep B inside boundaries (0 to 10)
        # self.b_x = np.clip(self.b_x, 0, 10)
        # self.b_y = np.clip(self.b_y, 0, 10)
        # self.b_z = np.clip(self.b_z, 0, 10)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:  
            self.b_y = min(10, self.b_y + 0.15)
        elif keys[pygame.K_s]:
            self.b_y = max(0, self.b_y - 0.15)
        
        if keys[pygame.K_d]:
            self.b_x = min(10, self.b_x + 0.15)
        elif keys[pygame.K_a]:
            self.b_x = max(0, self.b_x - 0.15)

        if keys[pygame.K_UP]:
            self.b_z = min(10, self.b_z + 0.15)
        elif keys[pygame.K_DOWN]:
            self.b_z = max(0, self.b_z - 0.15)

    def game_over(self):
        rel_x = self.a_x - self.b_x
        rel_y = self.a_y - self.b_y
        rel_z = self.a_z - self.b_z
        distance = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
        return distance < 0.2
        

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
    

# env = DoubleDroneGym()
# obs = env.reset()
# while True:
#     # Take a random action
#     currStep = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(currStep)

    

#     # Render the game
#     env.render('human')

#     if done == True:
#         break


# env.close()