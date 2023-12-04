import os
from datetime import datetime
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
run_name = os.environ["run_name"]


#####-----------------CONFIG-----------------#####
TEST = os.environ.get("TEST", True)
if type(TEST) != bool:
    TEST = TEST == "True"

## Hyperparameters
num_envs = 28 if not TEST else 1
batch_size = 512
n_steps = batch_size * 10
episode_length = n_steps * 24
total_timesteps = num_envs * episode_length
nb_epochs = 10
## Hyperparameters


#####-----------------CONFIG-----------------#####

if __name__ == "__main__":
    save_path = f"trained/PKM_{run_name}"
    configs = {
        # "rom_path": "ROMs/Pokemon Red.gb",
        "render_mode": "rgb_array" if not TEST else "human",
        "verbose": False if not TEST else True,
        "max_steps": 25000,
        "save_video": True,
        "save_screens": False,
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
    # env = VecVideoRecorder(
    #     env,
    #     _get_path("vec_videos"),
    #     record_video_trigger=lambda x: x % 2000 == 0,
    #     video_length=200,
    # )
    # eval_env = SubprocVecEnv([lambda: make_env(**configs) for _ in range(1)])

    model = PPO(
        "MultiInputPolicy",
        env,
        device="cuda",
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=configs.get("gamma"),
        n_epochs=configs.get("n_epochs"),
        ent_coef=configs.get("ent_coef"),
        learning_rate=configs.get("learning_rate"),
        vf_coef=configs.get("vf_coef"),
        # verbose=1,
        tensorboard_log=_get_path("tensorboard"),
    )

    for episode in range(nb_epochs):
        print_section(
            f"RUN NAME: {os.environ['run_name']}  EPISODE: {episode} of {nb_epochs}"
        )

        run = wandb.init(
            # set the wandb project where this run will be logged
            project="pkm-rl",
            sync_tensorboard=True,
            group=os.environ["run_name"],
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
