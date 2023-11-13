from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from utils import create_env
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0


def print_section(text):
    print("" * 80)
    print("=" * 80)
    print(text)
    print("=" * 80)
    print("" * 80)


## TEST

num_envs = 1
render_mode = "human"
nb_episodes = 5
timesteps_per_env = 100
verbose = False


## TRAIN

# num_envs = 12
# render_mode = None
# nb_epochs = 250
# timesteps_per_epoch = 5000
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
