from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from utils import create_env
import os


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
    num_envs = 10
    render_mode = None
    options = {
        "evaluate_rewards": True,
        # "save_video": True,
    }
    for i in range(20):
        print(f"Training model RUN {i+1}")
        env = SubprocVecEnv(
            [
                lambda: create_env(render_mode=render_mode, **options)
                for _ in range(num_envs)
            ]
        )
        ## if file named trained/PKM exists, load it
        if os.path.isfile("trained/PKM.zip"):
            model = PPO.load("trained/PKM", env=env, device="cuda")
        else:
            model = PPO("CnnPolicy", env, device="cuda")
        total_timesteps = 50000  # You can adjust this as needed
        model.learn(total_timesteps)
        model.save("trained/PKM")
        env.close()
