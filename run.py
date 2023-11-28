import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from utils.PkmEnv import PkmEnv
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
configs = {
    "rom_path": "ROMs/Pokemon Red.gb",
    "render_mode": "human",
    "emulation_speed": 1,
    "verbose": True,
    "verbose_exclude": [],
    "max_progress_without_reward": 999999,
    "log_type": "run",
    "run_id": run_id,
    "max_level_threshold": 8,
    "save_video": True,
}
#####-----------------CONFIG-----------------#####


def create_env(**configs):
    env = PkmEnv(**configs)
    seed = random.getrandbits(64)
    env.reset(seed=seed)
    return env


def get_model(model_path, env):
    batch_size = 500
    model_params = dict(
        env=env,
        device="cuda",
        batch_size=batch_size,
        n_steps=batch_size * 10,
        gamma=0.998,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=0.0003,
        vf_coef=0.5,
    )
    model = PPO.load(model_path, **model_params)
    return model


if __name__ == "__main__":
    # Enjoy trained agent
    vec_env = DummyVecEnv(
        [
            create_env(
                **configs,
            )
        ]
    )
    model = get_model(f"PKM_2023-11-27_18-01-41.zip", vec_env)

    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
