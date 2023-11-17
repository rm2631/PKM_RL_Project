import os
from datetime import datetime
from utils import create_env, print_section
from utils.LoggingCallback import TensorboardCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback

# from stable_baselines3.common.vec_env import VecMonitor


################
TEST = True
################

TOTAL_TIMESTEPS_TO_ACHIEVE = (
    7800000  ## This is the target for about 8 hours of training
)

num_envs = 10  ## nb of logical cores
timesteps_per_env = 1000
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


def calc_required_space():
    REQ_SPACE = num_envs * timesteps_per_env * (3 * 72) * 80 * 3 * 4
    print(f"Required space: {REQ_SPACE} B")


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
        tensorboard_log=f"{log_path}/",
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
        print_section(f"Starting Episode {episode}")
        model.learn(
            timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"Episode_{episode}",
            callback=[
                TensorboardCallback(),
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
