# Drone Chase with Stable Baselines3

This project adapts the [Aerial Navigation Code](https://github.com/AtsushiSakai/PythonRobotics/tree/master/AerialNavigation/drone_3d_trajectory_following) by Daniel Ingram (daniel-s-ingram) from [PythonRobotics](atsushisakai.github.io/PythonRobotics/) to create a simple 3D envrionment with two quadrotors. One quadrotor (Target) is controlled by the player with keyboard controls, and the other (Chaser) is trained with [Stable Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)'s PPO algorithm. The goal is for the user to not get caught by the Chaser.  

## Modifications to PythonRobotics code
The PythonRobotics code contains a Quadrotor.py file that plots the position of a quadrotor within Matplotlib based on its position (x,y,z), roll, pitch, and yaw. Within the drone_3d_trajectory_following.py uses PID to calculate the needed roll, pitch, yaw, and thrust for the quadrotor to follow a pre-determined trajectory.

The Quadrotor.py file was modified to simulate two quadrotors, a and b, instead of one. To set up the Gym environment, the drone_3d_trajectory_following.py was adapted such that the yaw, pitch, roll, and thrust of drone a (chaser drone) was to be controlled by the RL agent, and the player could control the position (x, y, and z coordinates) of drone b (target drone) with keyboard controls.

## Setting up the enviornment
The simulation code from above was translated to a [Gymnasium](https://gymnasium.farama.org/) environment. 

**State Space:** Relative position of chaser vs target drone, velocity of chaser drone, orientation of chaser drone, thrust of chaser drone

`self.state = np.array([rel_x, rel_y, rel_z,self.x_vel, self.y_vel, self.z_vel, self.roll, self.pitch, self.yaw, self.thrust], dtype=np.float32)`

**Action Space:** Continuous vector to represent thurst, yaw, pitch, and roll of chaser drone. 

> Thrust ∈ [0.8, 1.5]

> Yaw, Pitch, Roll ∈ [-1, 1]

**Reward Function:** 
- Distance penalty of `-distance * 0.5`
- Velocity penalty of `-0.1 * norm of velocity` to avoid high speeds
- Close bonus of `10` if chaser is within 0.5 distance from target
- Boundary penalty of `-5` if chaser hits edge of boundary
- Approach reward - Dot product of relative position and velocity of chaser drone


## Training the Model
During training, the target drone is set to move in a random but controlled sinusoidal motion. PPO was used as the RL envrionment, and 660k timesteps was enough for reasonable chasing behavior. 



## Gameplay

