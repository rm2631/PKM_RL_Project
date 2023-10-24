from utils.grid_world import GridWorldEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines3 import A2C
env = GridWorldEnv()

RENDER_MODE = "human"

env = GridWorldEnv(render_mode=RENDER_MODE)
model = A2C.load(path=f"trained/{str(env)}", env=env, verbose=1)

obs, info = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}:", info)
    print(f"Action {i}:", action)
    print(f"Observation {i}:", obs)
    env.render(RENDER_MODE)


    