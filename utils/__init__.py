import random
from utils.PKM_env import PKM_env


def create_env(render_mode=None, evaluate_rewards=True):
    env = PKM_env(render_mode=render_mode, evaluate_rewards=evaluate_rewards)
    seed = random.getrandbits(128)
    env.reset(seed=seed)
    return env