# Drone Chase with Stable Baselines3

This project adapts the [Aerial Navigation Code](https://github.com/AtsushiSakai/PythonRobotics/tree/master/AerialNavigation/drone_3d_trajectory_following) by Daniel Ingram (daniel-s-ingram) from [PythonRobotics](atsushisakai.github.io/PythonRobotics/) to create a simple 3D envrionment with two quadrotors. One quadrotor (Target) is controlled by the player with keyboard controls, and the other (Chaser) is trained with [Stable Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)'s PPO algorithm. The goal is for the user to not get caught by the Chaser.  

## Modifications to PythonRobotics code
The PythonRobotics code contains a `Quadrotor.py` file that plots the position of a quadrotor within Matplotlib based on its position (x,y,z), roll, pitch, and yaw.`drone_3d_trajectory_following.py` uses PID to calculate the needed roll, pitch, yaw, and thrust for the quadrotor to follow a pre-determined trajectory.

`Quadrotor.py` was modified to become `MultipleQuad.py` simulate two quadrotors, Chaser and Target, instead of one. To set up the Gym environment, `drone_3d_trajectory_following.py` was adapted such that the yaw, pitch, roll, and thrust of Drone A (Chaser) was to be controlled by the RL agent, and the player could control the position (x, y, and z coordinates) of Drone B (Target) with keyboard controls.

## Setting up the enviornment
The simulation code from above was translated to a [Gymnasium](https://gymnasium.farama.org/) environment. 

**State Space:** Relative position of chaser vs target drone, velocity of chaser drone, orientation of chaser drone, thrust of chaser drone

`self.state = np.array([rel_x, rel_y, rel_z,self.x_vel, self.y_vel, self.z_vel, self.roll, self.pitch, self.yaw, self.thrust], dtype=np.float32)`

**Action Space:** Continuous vector to represent thurst, yaw, pitch, and roll of chaser drone. 

> Thrust ∈ [0.8, 1.5]

> Yaw, Pitch, Roll ∈ [-1, 1]

**Reward Function:** 
- Proximity reward - Reward decreases exponentially as the distance increases
- Approach reward - Dot product of relative position and velocity of chaser drone
- Velocity penalty - Penalizes extremely high speeds (although I think I will try removing this)



## Training the Model
During training, the Target drone randomly selects one of the following movement patterns:
1. Static
2. Evasive - Move away from Chaser drone
3. Circular motion

I think the main roadblock I'm facing right now is how to properly simulate a user's playing behavior automatically during the training process. These three movement patterns are more predictable than a human player, and it don't seem to be training the model well enough.

PPO was used as the RL algorithm, and honestly I lost track of how many timesteps I trained it for. 

## Gameplay
Running the gym environmnet with the game flag set to TRUE allows the user to control the 
