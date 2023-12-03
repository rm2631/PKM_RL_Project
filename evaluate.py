from gym import wrappers
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# from utils.PkmEnv import PkmEnv
from utils.PkmEnv2 import PkmEnv2

configs = {
    # "rom_path": "ROMs/Pokemon Red.gb",
    "render_mode": "rgb_array",
    "verbose": False,
    "save_video": False,
    "save_screens": False,
    # "emulation_speed": 10,
    "max_steps": 10000,
}


# Load the environment and the trained model
def make_env(**configs):
    def _init():
        env = PkmEnv2(**configs)
        env = Monitor(env)
        return env

    return _init()


if __name__ == "__main__":
    env = SubprocVecEnv([lambda: make_env(**configs) for _ in range(5)])
    MODEL_PATH = "artifacts/2023-12-02_22-34-19/models/model.zip"
    model = PPO.load(MODEL_PATH, env=env)
    result = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True, return_episode_rewards=True
    )
    pass
