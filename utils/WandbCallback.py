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
                    "step reward": sum(step_rewards),
                    "total reward": sum(total_rewards),
                    "average total reward": sum(total_rewards) / self.num_envs,
                }
            )

            # for env_index in reward_index:
            #     ## Get the image from the environment
            #     self.__log_images(env_index)
            # self.logger.dump(step=self.num_timesteps)
        return True

    def __log_images(self, env_index):
        try:
            image = self.training_env.env_method("render")[env_index]
            self.logger.record(
                "trajectory/image",
                Image(image, "HWC"),
                exclude=("stdout", "log", "json", "csv"),
            )
        except:
            print("Error rendering image")
