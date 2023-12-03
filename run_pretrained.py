from gym import wrappers
from stable_baselines3 import PPO

# from utils.PkmEnv import PkmEnv
from utils.PkmEnv2 import PkmEnv2

configs = {
    # "rom_path": "ROMs/Pokemon Red.gb",
    "render_mode": "human",
    "verbose": True,
    "log_wandb": False,
    "save_video": False,
    "save_screens": False,
}

# Load the environment and the trained model
env = PkmEnv2(**configs)
MODEL_PATH = "artifacts/2023-12-02_21-35-04/models/model.zip"
model = PPO.load(MODEL_PATH, env=env)

obs, info = env.reset()
while True:
    action, _state = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(int(action))
    if terminated or truncated:
        obs, info = env.reset()  # Reset the environment for a new episode
