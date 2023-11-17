import random
from utils.PKM_env import PKM_env
from utils.PkmEnv2 import PkmEnv


def create_env(**configs):
    env = PkmEnv(**configs)
    seed = random.getrandbits(128)
    env.reset(seed=seed)
    return env


def print_section(text):
    print("" * 80)
    print("=" * 80)
    print(text)
    print("=" * 80)
    print("" * 80)
