from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from utils import create_env
import os


def train_model(num_envs, render_mode):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
    env = DummyVecEnv([lambda: create_env(render_mode=render_mode) for _ in range(num_envs)])
    model = PPO("MlpPolicy", env, verbose=1, device='cuda')
    total_timesteps = 10000  # You can adjust this as needed
    model.learn(total_timesteps)
    model.save("trained/PKM")
    env.close()