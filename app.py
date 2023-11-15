from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from utils import create_env, print_section
import os
from datetime import datetime

from utils.LoggingCallback import TensorboardCallback

# from stable_baselines3.common.vec_env import VecMonitor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

################
TEST = False
################

num_envs = 12  ## nb of logical cores
nb_episodes = 130
timesteps_per_env = 10000
render_mode = None
verbose = False
save_model = True
log_type = "train"
# 12 envs, 10000 timesteps per env, 130 episodes = around 8 hours of training


if TEST:
    num_envs = 5
    nb_episodes = 10
    timesteps_per_env = 100
    render_mode = "human"
    verbose = False
    save_model = False
    log_type = "test"


timesteps = num_envs * timesteps_per_env


def main():
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"./logs/{log_type}/{run_id}"

    save_path = "trained/PKM"
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
    # env = VecMonitor(env, log_path)

    model_params = dict(
        env=env,
        device="cuda",
        tensorboard_log=f"{log_path}/",
        n_steps=timesteps_per_env,
        batch_size=timesteps,
    )

    if os.path.isfile(f"{save_path}.zip"):
        model = PPO.load(save_path, **model_params)
    else:
        model = PPO(
            "CnnPolicy",
            **model_params,
        )

    if TEST:
        print_section("STARTING TEST")
    for episode in range(nb_episodes):
        print_section(f"Starting Episode {episode}")
        model.learn(
            timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"Episode_{episode}",
            callback=[
                TensorboardCallback(),
                ProgressBarCallback(),
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
