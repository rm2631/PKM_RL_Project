import os
from math import floor, sqrt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import mediapy as media
import pandas as pd
from datetime import datetime
import random
from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from utils import _get_path


class PkmEnv2(Env):
    def __init__(self, **configs):
        self.configs = configs
        self.render_mode = configs.get("render_mode")
        self.valid_actions = [
            [WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP],
            [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
            [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
            [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B],
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))

        ### Screen space
        single_screen_size_downscale_ratio = 4
        self.original_screen_size = (144, 160, 1)
        self.scaled_screen_size = (
            self.original_screen_size[0] // single_screen_size_downscale_ratio,
            self.original_screen_size[1] // single_screen_size_downscale_ratio,
            self.original_screen_size[2],
        )
        self.nb_stacked_screens = 3
        self.stacked_screen_size = (
            self.scaled_screen_size[0],
            self.scaled_screen_size[1],
            self.scaled_screen_size[2] * self.nb_stacked_screens,
        )

        ### Position space
        self.position_history_length = 100

        ### Observation space
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(
                    low=0, high=255, shape=self.stacked_screen_size, dtype=np.uint8
                ),
                "position": spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8),
                "position_history": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.position_history_length, 3),
                    dtype=np.uint8,
                ),
            }
        )
        self.pyboy = PyBoy(
            "ROMs/Pokemon Red.gb",
            debugging=False,
            disable_input=False,
            window_type="headless" if self.render_mode != "human" else "SDL2",
        )
        self.screen = self.pyboy.botsupport_manager().screen()

    def render(self):
        screen = self._get_screen(full_resolution=True)
        return screen

    def _get_obs(self):
        observation = {
            "screen": self._get_screen_stack(),
            "position": self.position,
            "position_history": list(
                reversed(self.position_history[-self.position_history_length :])
            ),
        }
        return observation

    def step(self, action):
        self.current_step_count += 1
        self._tick_screen(action)
        self._update_game_state()
        self._add_video_frame()
        observation = self._get_obs()
        reward = self._handle_rewards()
        terminated = False
        truncated = self._truncate()
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        self.seed = seed
        self.reset_id = str(random.randint(0, 2**32 - 1))
        self._load_rom_state()
        self._initialize_game_state()
        self._initialize_video_writer()
        observation = self._get_obs()
        info = {}
        return observation, info

    def close(self):
        self._close_video_writer()

    def _truncate(self):
        if self.current_step_count >= self.configs.get("max_steps"):
            return True
        return False

    def _initialize_game_state(self):
        self.current_step_count = 0
        self.screen_history = [self._get_screen()] * self.nb_stacked_screens
        self.position = self._get_position()
        self.position_history = [np.zeros(3)] * self.position_history_length

    def _update_game_state(self):
        self.position = self._get_position()

    def _tick_screen(self, action):
        frames = 24
        self.pyboy._rendering(False)
        for frame in range(frames):
            if frame == 0:
                self.pyboy.send_input(self.valid_actions[action][0])
            elif frame == 8:
                self.pyboy.send_input(self.valid_actions[action][1])
            elif frame == (frames - 1):
                self.pyboy._rendering(True)
            self.pyboy.tick()

    def _load_rom_state(self):
        state_name = f"ROMs/1_Pokemon Red.gb.state"
        self.pyboy.load_state(open(state_name, "rb"))

    ### PATH

    def _get_file_name(self, extension):
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = datetime_str + extension
        return file_name

    ### VIDEO

    def _get_screen(self, full_resolution=False):
        screen = self.screen.screen_ndarray()
        ## Compress the screen to self.single_screen_size
        if not full_resolution:
            screen = resize(screen, self.scaled_screen_size, anti_aliasing=True)
        else:
            screen = resize(screen, self.original_screen_size, anti_aliasing=True)
        return screen

    def _get_screen_stack(self):
        current_screen = self._get_screen()
        self.screen_history.append(current_screen)
        self.screen_history = self.screen_history[-self.nb_stacked_screens :]
        assert len(self.screen_history) == self.nb_stacked_screens
        screen_history = self.screen_history.copy()
        screen_history.reverse()
        screen_stack = np.concatenate(screen_history, axis=2)
        return screen_stack

    def _initialize_video_writer(self):
        if self.configs.get("save_video"):
            path = _get_path("videos", reset_id=self.reset_id)
            file_name = self._get_file_name(".mp4")
            full_path = os.path.join(path, file_name)
            self.video_writer = media.VideoWriter(
                full_path, self.original_screen_size[:2], fps=60
            )
            self.video_writer.__enter__()

    def _add_video_frame(self):
        if self.configs.get("save_video"):
            screen = self._get_screen(full_resolution=True)[:, :, 0]
            self.video_writer.add_image(screen)

    def _close_video_writer(self):
        if self.configs.get("save_video"):
            self.video_writer.close()

    ##### GAME UTILS #####

    def _get_position(self):
        Y = self.pyboy.get_memory_value(0xD361)
        X = self.pyboy.get_memory_value(0xD362)
        M = self.pyboy.get_memory_value(0xD35E)
        current_position = np.array([Y, X, M])
        return current_position

    ##### Rewards #####

    def log_reward(weight=1):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                reward = func(self)
                reward = reward * weight
                if self.configs.get("verbose"):
                    if reward != 0:
                        print(f"===== {func.__name__}: {reward} =====")
                return reward

            return wrapper

        return decorator

    def _handle_rewards(self):
        rewards = {"new_coord": self._new_coord_reward()}
        reward = sum(rewards.values())
        return reward

    @log_reward(weight=1)
    def _new_coord_reward(self):
        position = self.position
        position_history = self.position_history[-self.position_history_length :]
        if not any([np.array_equal(position, pos) for pos in position_history]):
            self.position_history.append(position)
            return 1
        return 0
