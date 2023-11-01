from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from utils import create_env
import os

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
    render_mode = 'human'
    env = create_env(render_mode=render_mode)
    model = PPO.load("trained/PKM", env=env, verbose=1)
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")