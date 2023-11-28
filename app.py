import os
from datetime import datetime
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    ProgressBarCallback,
)
from wandb.integration.sb3 import WandbCallback
import wandb
from utils.PkmEnv import PkmEnv
import random


#####-----------------CONFIG-----------------#####
TEST = os.environ.get("TEST", True)
if type(TEST) != bool:
    TEST = TEST == "True"

num_envs = 12 if not TEST else 1  ## Number of processes to use
timesteps_per_env = 100000 if not TEST else 1000  ## Number of timesteps per process
nb_episodes = 20

render_mode = None if not TEST else "human"
verbose = False if not TEST else True
save_model = True if not TEST else False
log_type = "train" if not TEST else "test"
max_progress_without_reward = 5000

timesteps = num_envs * timesteps_per_env
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"trained/PKM_{run_id}"
configs = {
    "rom_path": "ROMs/Pokemon Red.gb",
    "render_mode": render_mode,
    "emulation_speed": 1,
    "verbose": verbose,
    "verbose_exclude": [],
    "max_progress_without_reward": max_progress_without_reward,
    "log_type": log_type,
    "run_id": run_id,
    "max_level_threshold": 8,
    "save_video": True,
}
#####-----------------CONFIG-----------------#####


def create_env(**configs):
    env = PkmEnv(**configs)
    seed = random.getrandbits(64)
    env.reset(seed=seed)
    return env


def print_section(text):
    print("" * 80)
    print("=" * 80)
    print(text)
    print("=" * 80)
    print("" * 80)


def handle_callbacks(is_test, run_id):
    if is_test:
        return []
    else:
        wandb.init(
            # set the wandb project where this run will be logged
            project="pokemon-rl",
            group=f"MASTER_{run_id}",
        )
        return [
            ProgressBarCallback(),
            WandbCallback(),
        ]


def get_model(model_path, env):
    batch_size = 500
    model_params = dict(
        env=env,
        device="cuda",
        batch_size=batch_size,
        n_steps=batch_size * 10,
        gamma=0.998,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=0.0003,
        vf_coef=0.5,
    )
    if os.path.isfile(model_path):
        model = PPO.load(save_path, **model_params)
    else:
        model = PPO(
            "MultiInputPolicy",
            **model_params,
        )
    return model


if __name__ == "__main__":
    env = SubprocVecEnv(
        [
            lambda: create_env(
                **configs,
            )
            for _ in range(num_envs)
        ]
    )
    model = get_model(f"{save_path}.zip", env)
    for episode in range(nb_episodes):
        print_section(f"RUN ID: {run_id} - EPISODE: {episode} of {nb_episodes}")
        model.learn(
            timesteps,
            callback=handle_callbacks(TEST, run_id),
        )
        if save_model:
            model.save(save_path)
    env.close()
    wandb.finish()
