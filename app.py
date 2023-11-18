import os
from datetime import datetime
from utils.WandbCallback import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
import wandb
from utils.PkmEnv2 import PkmEnv
import random


def create_env(**configs):
    env = PkmEnv(**configs)
    seed = random.getrandbits(128)
    env.reset(seed=seed)
    return env


def print_section(text):
    print("" * 80)
    print("=" * 80)
    print(text)
    print("=" * 80)
    print("" * 80)


if __name__ == "__main__":
    ################
    TEST = False
    TOTAL_TIMESTEPS_TO_ACHIEVE = (
        7800000  ## This is the target for about 8 hours of training
    )
    ################

    num_envs = 10 if not TEST else 1  ## Number of processes to use
    timesteps_per_env = 5000 if not TEST else 1000  ## Number of timesteps per process
    nb_episodes = TOTAL_TIMESTEPS_TO_ACHIEVE // (num_envs * timesteps_per_env)
    render_mode = None if not TEST else "human"
    verbose = False if not TEST else False
    save_model = True if not TEST else False
    log_type = "train" if not TEST else "test"

    timesteps = num_envs * timesteps_per_env
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_path = "trained/PKM"
    configs = {
        "rom_path": "ROMs/Pokemon Red.gb",
        "render_mode": render_mode,
        "emulation_speed": 5,
        "verbose": verbose,
        "max_progress_without_reward": 10000,
        "log_type": log_type,
        "run_id": run_id,
    }

    wandb.init(
        # set the wandb project where this run will be logged
        project="pokemon-rl",
        # track hyperparameters and run metadata
        config={
            **configs,
        },
    )

    env = SubprocVecEnv(
        [
            lambda: create_env(
                **configs,
            )
            for _ in range(num_envs)
        ]
    )

    model_params = dict(
        env=env,
        device="cuda",
        n_steps=timesteps_per_env,
        batch_size=timesteps,
    )

    if os.path.isfile(f"{save_path}.zip"):
        model = PPO.load(save_path, **model_params)
    else:
        model = PPO(
            "MultiInputPolicy",
            **model_params,
        )

    if TEST:
        print_section("STARTING TEST")
    for episode in range(nb_episodes):
        print_section(f"Starting Episode {episode} of {nb_episodes}")
        model.learn(
            timesteps,
            callback=[
                WandbCallback(),
                # ProgressBarCallback(),
            ],
        )
        if episode % 4 == 0:
            if save_model:
                model.save(save_path)
    if save_model:
        model.save(save_path)
    env.close()
    wandb.finish()
