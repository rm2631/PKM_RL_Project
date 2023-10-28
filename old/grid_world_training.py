from utils.GridWorldEnv import GridWorldEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import A2C



if __name__ == "__main__":
    env = SubprocVecEnv([lambda: GridWorldEnv() for _ in range(10)])
    model = A2C("MultiInputPolicy", env, verbose=1)
    
    # learn_steps = 10
    # for i in range(learn_steps):
    model.learn(total_timesteps=100000)
    model.save(path=f"trained/GridWorldEnv")