import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
import wandb
from utils.PkmEnv import PkmEnv
import random


def create_env(group="train", **configs):
    wandb.init(
        # set the wandb project where this run will be logged
        project="pokemon-rl",
        # track hyperparameters and run metadata
        config={
            **configs,
        },
        group=group,
    )
    configs.update(
        {
            "run_name": wandb.run,
        }
    )

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


if __name__ == "__main__":
    ################
    TEST = True
    TOTAL_TIMESTEPS_TO_ACHIEVE = (
        3600000  ## This is the target for about 4 hours of training
    )
    ################

    num_envs = 12 if not TEST else 2  ## Number of processes to use
    timesteps_per_env = 2500 if not TEST else 1000  ## Number of timesteps per process
    nb_episodes = TOTAL_TIMESTEPS_TO_ACHIEVE // (num_envs * timesteps_per_env)
    render_mode = None if not TEST else "human"
    verbose = False if not TEST else True
    save_model = True if not TEST else False
    log_type = "train" if not TEST else "test"
    max_progress_without_reward = 500 if not TEST else 50

    timesteps = num_envs * timesteps_per_env
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = "trained/PKM"
    configs = {
        "rom_path": "ROMs/Pokemon Red.gb",
        "render_mode": render_mode,
        "emulation_speed": 5,
        "verbose": verbose,
        "verbose_exclude": ["_handle_position_reward"],
        "max_progress_without_reward": max_progress_without_reward,
        "log_type": log_type,
        "run_id": run_id,
        "max_level_threshold": 8,
        "save_video": True,
    }

    epoch = 0
    while True:
        epoch += 1
        print_section("STARTING NEW EPOCH: " + str(epoch))
        env = SubprocVecEnv(
            [
                lambda: create_env(
                    group=run_id,
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
        callbacks = []
        if not TEST:
            callbacks.append(ProgressBarCallback())
        if TEST:
            print_section("STARTING TEST")
        for episode in range(nb_episodes):
            print_section(f"Starting Episode {episode} of {nb_episodes}")
            model.learn(
                timesteps,
                callback=callbacks,
            )
            if save_model:
                model.save(save_path)
        env.close()
