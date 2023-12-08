import os
from math import floor, sqrt
from pathlib import Path
from tabnanny import verbose
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
from utils import _get_path, convert_array_to_image


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
        single_screen_size_downscale_ratio = (
            self.configs.get("single_screen_size_downscale_ratio") or 4
        )
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
        self.relevant_game_locations = [
            41,  # VIRIDIAN_POKECENTER
            45,  # VIRIDIAN_GYM
            54,  # PEWTER_GYM
            58,  # PEWTER_POKECENTER
            64,  # CERULEAN_POKECENTER
            65,  # CERULEAN_GYM
            68,  # MT_MOON_POKECENTER
            89,  # VERMILION_POKECENTER
            92,  # VERMILION_GYM
            133,  # CELADON_POKECENTER
            134,  # CELADON_GYM
            154,  # FUCHSIA_POKECENTER
            157,  # FUCHSIA_GYM
            166,  # CINNABAR_GYM
            171,  # CINNABAR_POKECENTER
            178,  # SAFFRON_GYM
            182,  # SAFFRON_POKECENTER
            1,  # VIRIDIAN_CITY
            2,  # PEWTER_CITY
            3,  # CERULEAN_CITY
            4,  # LAVENDER_TOWN
            5,  # VERMILION_CITY
            6,  # CELADON_CITY
            7,  # FUCHSIA_CITY
            8,  # CINNABAR_ISLAND
            9,  # INDIGO_PLATEAU
            10,  # SAFFRON_CITY
        ]

        ### Observation space
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(
                    low=0, high=255, shape=self.stacked_screen_size, dtype=np.uint8
                ),
                "party_level": spaces.Box(low=0, high=255, shape=(6,), dtype=np.uint8),
                "party_xp": spaces.Box(low=0, high=255, shape=(6,), dtype=np.uint8),
                "party_hp": spaces.Box(low=0, high=255, shape=(6,), dtype=np.uint8),
                "party_type": spaces.Box(low=0, high=255, shape=(6,), dtype=np.uint8),
                "party_type2": spaces.Box(low=0, high=255, shape=(6,), dtype=np.uint8),
            }
        )
        self.pyboy = PyBoy(
            "ROMs/Pokemon Red.gb",
            debugging=False,
            disable_input=False,
            window_type="headless" if self.render_mode != "human" else "SDL2",
            verbose=False,
        )
        if self.configs.get("emulation_speed"):
            self.pyboy.set_emulation_speed(self.configs.get("emulation_speed"))
        self.screen = self.pyboy.botsupport_manager().screen()

    def render(self):
        screen = self._get_screen(full_resolution=True)
        return screen

    def _get_obs(self):
        observation = {
            "screen": self._get_screen_stack(),
            "party_level": self._get_party_level(),
            "party_xp": self._get_party_xp(),
            "party_hp": self._get_party_hp(),
            "party_type": self._get_party_type(first_type=True),
            "party_type2": self._get_party_type(first_type=False),
        }
        return observation

    def step(self, action):
        self._initialize_step()
        self._tick_screen(action)
        self._update_game_state()
        self._add_video_frame()
        observation = self._get_obs()
        reward = self._handle_rewards()
        terminated = False
        truncated = self._get_truncate_status()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _initialize_step(self):
        self.current_step_count += 1
        self.saved_screen = False
        self.saved_screen_stack = False

    def reset(self, seed=None):
        self.seed = seed
        self.reset_id = str(random.randint(0, 2**32 - 1))
        self._load_rom_state()
        self._initialize_game_state()
        self._initialize_video_writer()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def close(self):
        self._close_video_writer()

    def _get_info(self):
        info = {
            "reset_id": self.reset_id,
            "current_step_count": self.current_step_count,
            # "position": self.position,
            # "position_history": self.position_history,
            "rewarded_maps": self.rewarded_maps,
            "max_party_level": self.max_party_level,
            "current_badge_count": self.current_badge_count,
            "rewards": [
                getattr(self, attr) for attr in dir(self) if attr.startswith("rt_")
            ],
        }
        return info

    def _get_truncate_status(self):
        if self.current_step_count >= (self.configs.get("max_steps") or 25000):
            if self.configs.get("verbose"):
                print("===== Truncated =====")
            return True
        return False

    def _initialize_game_state(self):
        self.current_step_count = 0
        self.screen_history = [self._get_screen()] * self.nb_stacked_screens
        self.position = self._get_position()
        self.position_history = []
        self.rewarded_maps = []
        self.rewarded_important_maps = []
        self.max_party_level = 6
        self.current_badge_count = 0
        self.current_party_hp = self._get_party_hp()

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
        if not self.configs.get("state_id"):
            state_id = random.randint(1, 4)
        else:
            state_id = self.configs.get("state_id")
        state_name = f"ROMs/{state_id}_Pokemon Red.gb.state"
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

    def _save_screen(self):
        if not self.saved_screen:
            if self.configs.get("save_screens"):
                screen = self._get_screen(full_resolution=True)
                path = _get_path("screens", reset_id=self.reset_id)
                file_name = self._get_file_name(".png")
                full_path = os.path.join(path, file_name)
                img = convert_array_to_image(screen)
                img.save(full_path)
                self.saved_screen = True

    def _save_screen_stack(self):
        pass
        if not self.saved_screen_stack:
            if self.configs.get("save_screens"):
                screen = self._get_screen_stack()
                path = _get_path("screens_stack", reset_id=self.reset_id)
                file_name = self._get_file_name(".png")
                full_path = os.path.join(path, file_name)
                img = convert_array_to_image(screen)
                img.save(full_path)
                self.saved_screen_stack = True

    ##### GAME UTILS #####

    def _get_position(self):
        Y = self.pyboy.get_memory_value(0xD361)
        X = self.pyboy.get_memory_value(0xD362)
        M = self.pyboy.get_memory_value(0xD35E)
        current_position = np.array([Y, X, M])
        return current_position

    def _get_party_level(self):
        party_level = [
            self.pyboy.get_memory_value(i)
            for i in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return party_level

    def _get_party_xp(self):
        party_xp = [
            self.pyboy.get_memory_value(i)
            for i in [0xD17B, 0xD1A7, 0xD1D3, 0xD1FF, 0xD22B, 0xD257]
        ]
        return party_xp

    def _get_party_hp(self):
        party_hp = [
            self.pyboy.get_memory_value(i)
            for i in [0xD16D, 0xD199, 0xD1C5, 0xD1F1, 0xD21D, 0xD249]
        ]
        return party_hp

    def _get_party_type(self, first_type=True):
        first_type = [0xD170, 0xD19C, 0xD1C8, 0xD1F4, 0xD220, 0xD24C]
        second_type = [0xD171, 0xD19D, 0xD1C9, 0xD1F5, 0xD221, 0xD24D]
        if first_type:
            type_array = first_type
        else:
            type_array = second_type
        party_type = [self.pyboy.get_memory_value(i) for i in type_array]
        return party_type

    ##### Rewards #####

    def log_reward(weight=1):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                func_name = func.__name__
                total_func_name = "rt_" + func_name
                if not hasattr(self, total_func_name):
                    setattr(self, total_func_name, 0)

                reward = func(self)
                reward = reward * weight
                if self.configs.get("verbose"):
                    if reward != 0:
                        print(f"===== {func_name}: {reward} =====")
                if reward != 0:
                    ## Keep track of the total reward
                    current_reward = getattr(self, total_func_name)
                    setattr(self, total_func_name, current_reward + reward)
                    self._save_screen()
                    self._save_screen_stack()
                return reward

            return wrapper

        return decorator

    def _handle_rewards(self):
        rewards = {
            "new_coord": self._reward_new_coord(),
            "new_map": self._reward_new_map(),
            "new_level": self._reward_new_level(),
            "new_badge": self._reward_new_badge(),
            "downed_pkm": self._reward_downed_pkm(),
        }
        reward = sum(rewards.values())
        return reward

    @log_reward(weight=0.0001)
    def _reward_new_coord(self):
        position = self.position
        if not any([np.array_equal(position, pos) for pos in self.position_history]):
            self.position_history.append(position)
            return 1
        return 0

    @log_reward(weight=2)
    def _reward_new_map(self):
        current_map = self.position[2]
        if not current_map in self.rewarded_maps:
            self.rewarded_maps.append(current_map)
            return 1
        return 0

    @log_reward(weight=1)
    def _reward_relevant_map(self):
        current_map = self.position[2]
        if current_map in self.relevant_game_locations:
            if not current_map in self.rewarded_important_maps:
                self.rewarded_important_maps.append(current_map)
                return 1
        return 0

    @log_reward(weight=0.5)
    def _reward_new_level(self):
        """
        Reward for leveling up or catching a new pokemon.
        """
        party_level = self._get_party_level()
        current_max_party_level = sum(party_level)
        if current_max_party_level <= self.max_party_level:
            return 0
        self.max_party_level = current_max_party_level
        return 1

    @log_reward(weight=5)
    def _reward_new_badge(self):
        """
        Reward for getting a new badge.
        """
        self.previous_badge_count = self.current_badge_count
        self.current_badge_count = bin(self.pyboy.get_memory_value(0xD356)).count("1")
        if self.current_badge_count > self.previous_badge_count:
            print(f"{self.reset_id}: New badge!")
            return 1
        return 0

    @log_reward(weight=-0.1)
    def _reward_downed_pkm(self):
        self.previous_party_hp = self.current_party_hp
        self.current_party_hp = self._get_party_hp()
        ##Check if a pokemon has been downed
        for prev, curr in zip(self.previous_party_hp, self.current_party_hp):
            if prev != 0 and curr <= 0:
                return 1
        return 0
