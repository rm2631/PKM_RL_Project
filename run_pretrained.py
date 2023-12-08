from os import truncate
from gym import wrappers
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from utils import _get_path, save_log, set_run_name
from stable_baselines3.common.env_util import make_vec_env


# from utils.PkmEnv import PkmEnv
from utils.PkmEnv2 import PkmEnv2

VIEW = False

num_envs = 16 if not VIEW else 1

configs = {
    "render_mode": "rgb_array" if not VIEW else "human",
    "single_screen_size_downscale_ratio": 4,
    "verbose": False,
    "max_steps": 100,
    "save_video": True,
    "save_screens": False,
    # "emulation_speed": 10,
}


# Load the environment and the trained model
def make_env(**configs):
    def _init():
        env = PkmEnv2(**configs)
        env = Monitor(env)
        return env

    return _init()


if __name__ == "__main__":
    model_name = "2023-12-06_17-55-08"
    MODEL_PATH = f"artifacts/{model_name}/models/model.zip"
    run_name = model_name + "_test"
    run_name = set_run_name(run_name)

    env = SubprocVecEnv([lambda: make_env(**configs) for _ in range(num_envs)])
    model = PPO.load(MODEL_PATH)
    obs = env.reset()

    test_done = False
    while not test_done:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)
        if any(done):
            test_done = True
    log_path = _get_path("test_logs")
    save_log(log_path + "/log.json", info)
    env.close()
