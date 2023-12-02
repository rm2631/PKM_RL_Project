from gym import wrappers
from stable_baselines3 import PPO
from utils.PkmEnv import PkmEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

MODEL_PATH = "trained/PKM_2023-11-30_22-56-02.zip"

configs = {
    "rom_path": "ROMs/Pokemon Red.gb",
    "render_mode": "human",
    "emulation_speed": 1,
    "verbose": True,
    "verbose_exclude": [],
    "episode_length": 500,
    "log_type": "test",
    "run_id": "test",
    "max_level_threshold": 8,
    "save_video": False,
    "log_wandb": False,
    "force_initial_state": 1,
}


def create_env(**configs):
    env = PkmEnv(**configs)
    return env


# Load the environment and the trained model
env = create_env(**configs)
model = PPO.load(MODEL_PATH)


obs, info = env.reset()
while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
