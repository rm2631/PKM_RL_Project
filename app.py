import os
from datetime import datetime
from utils import create_env, print_section
from utils.LoggingCallback import TensorboardCallback
from utils.WandbCallback import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
import wandb

# from stable_baselines3.common.vec_env import VecMonitor


################
TEST = True
################

TOTAL_TIMESTEPS_TO_ACHIEVE = (
    7800000  ## This is the target for about 8 hours of training
)

num_envs = 10  ## nb of logical cores
timesteps_per_env = 5000  ## nb of timesteps per logical core
nb_episodes = TOTAL_TIMESTEPS_TO_ACHIEVE // (num_envs * timesteps_per_env)
render_mode = None
verbose = False
save_model = True
log_type = "train"


if TEST:
    num_envs = 2
    timesteps_per_env = 1000
    nb_episodes = 50
    render_mode = "human"
    verbose = True
    save_model = False
    log_type = "test"


timesteps = num_envs * timesteps_per_env


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"./logs/{log_type}/{run_id}"

    save_path = "trained/PKM"
    configs = {
        "rom_path": "ROMs/Pokemon Red.gb",
        "render_mode": render_mode,
        "emulation_speed": 5,
        "verbose": verbose,
        "max_progress_without_reward": 10000,
    }

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

    wandb.init(
        # set the wandb project where this run will be logged
        project="pokemon-rl",
        # track hyperparameters and run metadata
        config={
            "num_envs": num_envs,
            "timesteps_per_env": timesteps_per_env,
            "nb_episodes": nb_episodes,
            "log_type": log_type,
            **model_params,
        },
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


if __name__ == "__main__":
    main()
    # calc_required_space()
