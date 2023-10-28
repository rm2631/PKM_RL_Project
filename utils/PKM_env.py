import numpy as np
from pyboy import PyBoy
import pandas as pd
from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import os
import pyboy.openai_gym as gym
import random

def _log_duration(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print(f"{func.__name__} took {end - start}")
        return result
    return wrapper

class PKM_env(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    command_map = {
            1: [WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP],
            2: [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
            3: [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
            4: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
            5: [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
            6: [WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B],
        }

    def __init__(self, render_mode=None, **kwargs):
        self.render_mode = render_mode
        window_type = 'headless' if render_mode != 'human' else 'SDL2'
        self.pyboy = PyBoy(
            'ROMs/Pokemon Red.gb',
            window_type=window_type,
        )
        self.instance = random.getrandbits(128)
        self.screen = self.pyboy.botsupport_manager().screen()
        self.previous_screens = []
        self.action_space = spaces.Discrete(len(self.command_map))
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=int)
        self.evaluate_rewards = kwargs.get("evaluate_rewards", False)

    def step(
        self, action
    ) :
        self._send_command(action)
        obs = self._get_obs()
        reward = self._handle_reward(obs)
        terminated = False
        info = {}
        return obs, reward, terminated, False, info
    
    def reset(
        self,
        seed=None,
        options=None,
    ):  # type: ignore
        super().reset(seed=seed, options=options)
        self.seed = seed
        self.pyboy.load_state(open("ROMs/Pokemon Red.gb.state", "rb"))
        obs = self._get_obs()
        info = {}
        return obs, info

    def render(self):
        if self.render_mode == "human":
            return None

    def close(self):
        self.pyboy.stop()

    def _handle_reward(self, obs):
        if not self.evaluate_rewards:
            return 0
        new_reward = 0
        new_reward += self._handle_screen_similarity(obs)
        new_reward += self._handle_successfull_attack(obs)
        return new_reward
    
    def _handle_successfull_attack(self, obs):
        
        return 0

    def _handle_screen_similarity(self, obs):
        obs = np.squeeze(obs)
        similarities = [ssim(screen, obs, multichannel=True, channel_axis=2) for screen in self.previous_screens]
        if len(similarities) == 0 or max(similarities) <= 0.50:
            self.previous_screens.append(obs)
            self._save_screen()
            return 1
        return 0

    def _get_obs(self):
        game_pixels_render = self.screen.screen_ndarray()
        return game_pixels_render
    
    def _get_info(self):
        return {}
    
    def _send_command(self, command):
        if command in self.command_map:
            self.pyboy.send_input(self.command_map[command][0])
            self.pyboy.tick()
            self.pyboy.send_input(self.command_map[command][1])

    def _save_screen(self):
        now = datetime.now()
        day_string = now.strftime("%d-%m-%Y")
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        directory = f"screenshots/{day_string}/{self.instance}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        pil_image = self.pyboy.screen_image()
        #save image with datetime as name
        pil_image.save(f"{directory}/{dt_string}.png")

