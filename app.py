from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from utils import create_env, print_section
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

## TEST

num_envs = 1
nb_episodes = 5
timesteps_per_env = 100
render_mode = "human"
verbose = False


## TRAIN

# num_envs = 12 ## nb of logical cores
# nb_episodes = 250
# timesteps_per_env = 5000
# render_mode = None
# verbose = False


timesteps = num_envs * timesteps_per_env


def main():
    save_path = "trained/"
    options = {
        "evaluate_rewards": True,
        "verbose": verbose,
    }

    env = SubprocVecEnv(
        [
            lambda: create_env(
                render_mode=render_mode,
                **options,
            )
            for _ in range(num_envs)
        ]
    )

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_params = dict(
        env=env,
        device="cuda",
        tensorboard_log=f"./logs/{run_id}/",
        n_steps=timesteps_per_env,
    )

    if os.path.isfile("trained/PKM.zip"):
        model = PPO.load("trained/PKM", **model_params)
    else:
        model = PPO(
            "CnnPolicy",
            **model_params,
        )

    for episode in range(nb_episodes):
        print_section(f"Starting Episode {episode}")
        model.learn(
            timesteps,
            callback=[
                ProgressBarCallback(),
            ],
        )
    model.save(f"{save_path}/PKM.zip")
    env.close()


if __name__ == "__main__":
    main()
