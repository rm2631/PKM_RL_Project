from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.logger import Image
import wandb


class WandbCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.num_envs = None

    def _on_step(self) -> bool:
        if self.num_envs is None:
            self.num_envs = self.model.n_envs

        total_rewards = self.training_env.get_attr("total_rewards")

        step_rewards = self.locals.get("rewards")
        reward_index = [
            index for index, i in enumerate(self.locals.get("rewards")) if i != 0
        ]
        if len(reward_index) != 0 or self.num_timesteps % 1000 == 0:
            wandb.log(
                {
                    "average total reward": sum(total_rewards) / self.num_envs,
                }
            )
            for env_index in reward_index:
                ## Get the image from the environment
                self.__log_images(env_index)
            # self.logger.dump(step=self.num_timesteps)
            for env_info in self.locals.get("infos"):
                reward_memory = env_info.get("reward_memory")
                if reward_memory is not None:
                    [wandb.log({key: value}) for key, value in reward_memory.items()]
                    total_reward = sum(reward_memory.values())
                    if total_reward != 0:
                        wandb.log({"step reward": total_reward})
        ##TODO: Add a log when an environment resets
        return True

    def __log_images(self, env_index, caption="Screenshot"):
        try:
            image_array = self.training_env.env_method("render")[env_index]
            image_array = image_array[:, :, -1]
            image = wandb.Image(image_array, caption=caption)
            wandb.log({"screenshot": image})
        except:
            print("Error rendering image")
