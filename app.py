import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from utils.PkmEnv import PkmEnv
from stable_baselines3.common.evaluation import evaluate_policy
from utils import print_section, handle_callbacks


#####-----------------CONFIG-----------------#####
TEST = os.environ.get("TEST", True)
if type(TEST) != bool:
    TEST = TEST == "True"

## Hyperparameters
num_envs = 24 if not TEST else 1  ## Number of processes to use
batch_size = 512
n_steps = batch_size * 10  ## 5120
episode_length = n_steps * 16  ## 81920
timesteps_per_env = episode_length * 12  ## 983040
total_timesteps = num_envs * timesteps_per_env  ## 23592960
## Hyperparameters

nb_epochs = 10

render_mode = None if not TEST else "human"
verbose = False if not TEST else True
save_model = True if not TEST else False
log_type = "train" if not TEST else "test"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"trained/PKM_{run_id}"


#####-----------------CONFIG-----------------#####


def create_env(**configs):
    env = PkmEnv(**configs)
    return env


if __name__ == "__main__":
    configs = {
        "rom_path": "ROMs/Pokemon Red.gb",
        "render_mode": render_mode,
        "emulation_speed": 1,
        "verbose": verbose,
        "verbose_exclude": [],
        "episode_length": episode_length,
        "log_type": log_type,
        "run_id": run_id,
        "save_video": True,
        "log_wandb": True,
        ## Hyperparameters
        "num_envs": num_envs,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "episode_length": episode_length,
        "timesteps_per_env": timesteps_per_env,
        "total_timesteps": total_timesteps,
        "gamma": 0.998,
        "n_epochs": 1,
    }
    env = SubprocVecEnv(
        [
            lambda: create_env(
                **configs,
            )
            for _ in range(num_envs)
        ]
    )

    model = PPO(
        "MultiInputPolicy",
        env=env,
        device="cuda",
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=configs.get("gamma"),
        n_epochs=configs.get("n_epochs"),
    )
    for episode in range(nb_epochs):
        print_section(f"RUN ID: {run_id} - EPISODE: {episode} of {nb_epochs}")
        model.learn(
            total_timesteps,
            callback=handle_callbacks(TEST),
        )
        model.save(save_path)
    env.close()
