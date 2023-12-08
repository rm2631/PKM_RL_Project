import os
from datetime import date, datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.PkmEnv2 import PkmEnv2
from utils import print_section, _get_path
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import (
    ProgressBarCallback,
    EvalCallback,
)

#####-----------------GLOBAL-----------------#####
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
os.environ["run_name"] = os.environ.get("run_name") or datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)

#####-----------------CONFIG-----------------#####
TEST = os.environ.get("TEST", True)
if type(TEST) != bool:
    TEST = TEST == "True"

## Hyperparameters
num_envs = 18 if not TEST else 1
batch_size = 512
n_steps = batch_size * 10
episode_length = n_steps * 24
total_timesteps = num_envs * episode_length
nb_epochs = 20
## Hyperparameters


#####-----------------CONFIG-----------------#####

if __name__ == "__main__":
    configs = {
        # "rom_path": "ROMs/Pokemon Red.gb",
        "render_mode": "rgb_array" if not TEST else "human",
        "single_screen_size_downscale_ratio": 4 if not TEST else 1,
        "verbose": False if not TEST else True,
        "max_steps": 25000,
        "save_video": True,
        "save_screens": False if not TEST else True,
        "gamma": 0.998,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "learning_rate": 0.0003,
        "vf_coef": 0.5,
    }

    def make_env(**configs):
        def _init():
            env = PkmEnv2(**configs)
            env = Monitor(env)
            return env

        return _init()

    env = SubprocVecEnv([lambda: make_env(**configs) for _ in range(num_envs)])

    model_config = {
        "device": "cuda",
        "batch_size": batch_size,
        "n_steps": n_steps,
        "gamma": configs.get("gamma"),
        "n_epochs": configs.get("n_epochs"),
        "ent_coef": configs.get("ent_coef"),
        "learning_rate": configs.get("learning_rate"),
        "vf_coef": configs.get("vf_coef"),
        "tensorboard_log": _get_path("tensorboard"),
    }

    model = PPO(
        "MultiInputPolicy",
        env,
        **model_config,
    )

    for episode in range(nb_epochs):
        print_section(
            f"RUN NAME: {os.environ['run_name']}  EPISODE: {episode} of {nb_epochs}"
        )
        run = wandb.init(
            project="pkm-rl",
            sync_tensorboard=True,
            group=os.environ["run_name"],
            config={
                "test_run": TEST,
                **configs,
            },
        )

        callbacks = [
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=_get_path("models"),
                model_save_freq=1000,
                # verbose=2,
            ),
        ]

        if not TEST:
            callbacks.append(ProgressBarCallback())

        model.learn(
            total_timesteps,
            callback=callbacks,
        )
        run.finish()
    env.close()
