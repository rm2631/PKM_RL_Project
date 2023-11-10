from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from utils import create_env
import os

## TEST

num_envs = 1
render_mode = "human"
nb_epochs = 2
total_timesteps = 5


## TRAIN

# num_envs = 5
# render_mode = "human"
# nb_epochs = 2
# total_timesteps = 20000


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
    options = {
        "evaluate_rewards": True,
        # "save_video": True,
    }
    for epoch in range(nb_epochs):
        ## Print a big header for each run
        print("" * 80)
        print("=" * 80)
        print(f"Training Epoch {epoch}")
        print("=" * 80)
        print("" * 80)

        env = SubprocVecEnv(
            [
                lambda: create_env(
                    render_mode=render_mode, epoch=epoch, env_index=env_i, **options
                )
                for env_i in range(num_envs)
            ]
        )
        ## if file named trained/PKM exists, load it
        if os.path.isfile("trained/PKM.zip"):
            model = PPO.load("trained/PKM", env=env, device="cuda")
        else:
            model = PPO("CnnPolicy", env, device="cuda")
        model.learn(total_timesteps)
        model.save("trained/PKM")
        env.close()


if __name__ == "__main__":
    main()
