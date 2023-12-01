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

num_envs = 12 if not TEST else 2  ## Number of processes to use
timesteps_per_env = 100000 if not TEST else 25000  ## Number of timesteps per process
nb_epochs = 20

render_mode = None if not TEST else "human"
verbose = False if not TEST else True
save_model = True if not TEST else False
log_type = "train" if not TEST else "test"
episode_length = timesteps_per_env // 5

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
    "episode_length": episode_length,
    "log_type": log_type,
    "run_id": run_id,
    "max_level_threshold": 7,
    "save_video": True,
    "log_wandb": True,
}
#####-----------------CONFIG-----------------#####


def create_env(**configs):
    env = PkmEnv(**configs)
    return env


if __name__ == "__main__":
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
        batch_size=500,
        n_steps=500 * 80,
        gamma=0.998,
        n_epochs=1,
    )
    for episode in range(nb_epochs):
        print_section(f"RUN ID: {run_id} - EPISODE: {episode} of {nb_epochs}")
        model.learn(
            timesteps,
            callback=handle_callbacks(TEST),
        )
        model.save(save_path)
    env.close()
