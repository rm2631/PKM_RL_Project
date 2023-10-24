from utils.GridWorldEnv import GridWorldEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np

from stable_baselines3 import A2C
env = GridWorldEnv()

RENDER_MODE = "human"
USE_TRAINING = True

env = GridWorldEnv(render_mode=RENDER_MODE)
if USE_TRAINING:
    model = A2C.load(path=f"trained/{str(env)}", env=env, verbose=1)
else:
    model = A2C("MultiInputPolicy", env, verbose=1)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    if USE_TRAINING:
        action, _state = model.predict(obs, deterministic=True)
    else:
        action = np.array([env.action_space.sample()])
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")