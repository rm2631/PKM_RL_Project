from utils.grid_world import GridWorldEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines3 import A2C
env = GridWorldEnv()

model = A2C("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
model.save(path=f"trained/{str(env)}")