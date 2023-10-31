from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from utils import create_env
import os

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
    render_mode = 'human'
    env = create_env(render_mode=render_mode)
    model = PPO.load("trained/PKM", env=env, verbose=1, device='cpu')
    total_timesteps = 10000 # You can adjust this as needed
    model.learn(total_timesteps)
    env.close()