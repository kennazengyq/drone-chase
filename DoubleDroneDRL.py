import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from DoubleDroneGym import DoubleDroneGym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure


import os

def train_sb3():
    env = DoubleDroneGym(game=False)
    check_env(env, warn=True)
    logdir = 'logs'
    model_dir = 'again'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    modelPPO = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    modelPPO = PPO.load('mid.zip')

    modelPPO.set_env(env)



    TIMESTEPS = 50000
    iter = 1
    while iter <= 200:
        tb_log_name=f'model_iter_{iter}'
        modelPPO.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True, tb_log_name=tb_log_name)
        modelPPO.save(f"{model_dir}/ppo{TIMESTEPS * iter}")
        iter += 1

def test_sb3():
    env = DoubleDroneGym(game=False)
    model = PPO.load('mid.zip')
    obs = env.reset()[0]

    while True:
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        env.render()

        if done:
            break

    env.close()

if __name__=="__main__":
    train_sb3()
    

# episodes = 10
# for ep in range(episodes):
#     print("running simulations post learning")
#     obs, info = env.reset()
#     print('obs', obs)
#     done = False
#     while not done:
#         env.render()
#         action, _states = model.predict(obs)
#         obs, reward, done, truncated, info = env.step(action)

# env.close()