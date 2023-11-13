import random
from utils.PKM_env import PKM_env


def create_env(render_mode=None, **options):
    env = PKM_env(render_mode=render_mode, **options)
    seed = random.getrandbits(128)
    env.reset(seed=seed)
    return env


def print_section(text):
    print("" * 80)
    print("=" * 80)
    print(text)
    print("=" * 80)
    print("" * 80)
