import gymnasium as gym
from utils import create_env, print_section
from stable_baselines3 import PPO


render_mode = "human"
save_path = "trained/PKM"

env = create_env(
    render_mode=render_mode,
)

model_params = dict(
    env=env,
)

model = PPO.load(save_path, **model_params)

obs = env.reset()
while True:
    done = False
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # env.render("human")
