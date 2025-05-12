# Drone Chase with Stable Baselines3

This project adapts the [Aerial Navigation Code](https://github.com/AtsushiSakai/PythonRobotics/tree/master/AerialNavigation/drone_3d_trajectory_following) by Daniel Ingram (daniel-s-ingram) from [PythonRobotics](atsushisakai.github.io/PythonRobotics/) to create a simple 3D envrionment with two quadrotors. One quadrotor (Target) is controlled by the player with keyboard controls, and the other (Chaser) is trained with [Stable Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)'s PPO algorithm. The goal is for the user to not get caught by the Chaser.  

## Modifications to PythonRobotics code
The PythonRobotics code contains a `Quadrotor.py` file that plots the position of a quadrotor within Matplotlib based on its position (x,y,z), roll, pitch, and yaw.`drone_3d_trajectory_following.py` uses PID to calculate the needed roll, pitch, yaw, and thrust for the quadrotor to follow a pre-determined trajectory.

`Quadrotor.py` was modified to become `MultipleQuad.py` simulate two quadrotors, Chaser and Target, instead of one. To set up the Gym environment, `drone_3d_trajectory_following.py` was adapted such that the yaw, pitch, roll, and thrust of Drone A (Chaser) was to be controlled by the RL agent, and the player could control the position (x, y, and z coordinates) of Drone B (Target) with keyboard controls.

## Setting up the environment
The simulation code from above was translated to a [Gymnasium](https://gymnasium.farama.org/) environment. 

**State Space:** 

Relative position of Chaser vs Target drone, Velocity of Chaser drone, Orientation of Chaser drone, Thrust of Chaser drone

`self.state = np.array([rel_x, rel_y, rel_z,self.x_vel, self.y_vel, self.z_vel, self.roll, self.pitch, self.yaw, self.thrust], dtype=np.float32)`

**Action Space:** Continuous vector to represent thurst, yaw, pitch, and roll of chaser drone. 

> Thrust ∈ [0.8, 2.0]

> Yaw, Pitch, Roll ∈ [-1, 1]

**Reward Function:** 

This is constantly being tweaked, with specifics defined in the `calc_reward()` function, but some components I've tried are:
- Distance Reward: Inverse relationship to distance; Smaller distance = greater reward
- Collision Reward: Large bonus reward if distance < collision threshold
- Approach Reward: Gives bonus if Chaser's velocity is approaching Target



## Training the Model
During training, the Target drone randomly selects one of the following movement patterns:
1. Static
2. Evasive - Move away from Chaser drone
3. Circular Motion - Radius and speed is randomized; Also contains noise

StableBaselines3's PPO model was used as the RL algorithm. 

## Current Performance

I've been messing around with the reward function, model parameters, and action space for a while, and I would train the models for timesteps in the upper hundreds of thousands range. Overall the performance has been pretty mediocre. There is some semblance of "catching" behavior, but the Chaser drone can be quite unstable and is too slow and imprecise. 

![chasing](https://github.com/user-attachments/assets/21df01d2-3711-4e20-be84-63bb8006d81f)

_Blue = User-controlled, Red = models2/ppo700000.zip_


The main issue is that the Chaser would often end up hovering near the Target instead of actually colliding with it. 

![hover](https://github.com/user-attachments/assets/6688b22e-d348-4eda-ba4e-c4d236fdfafc)


## Current Plan
Here are a few things I'm working on right now in hopes of improving the model:
- • Add penalty in reward function for "social distancing" behavior - maybe penalize hovering and increase bonus for collision threshold
- • Normalizing reward values - the files `NDoubleDroneGym.py` and `NDoubleDroneDRL.py` are copies of my original environment, except the reward and states are normalized. My loss and value loss were huge with the original environment, so hopefully this will provide some stability. 
- • Improving simulation of user actions - The evasive behavior mode tends to drive the target drone into edges and corners. I also don't want the RL agent to purely learn how to move in a sinusoidal shape.
- • I should probably learn how to better read and understand the logs and loss graphs

