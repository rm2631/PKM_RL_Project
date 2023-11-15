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
from utils.numpy_array_to_image_and_save import numpy_array_to_image
from utils.reward_functions import (
    handle_hp_change_reward,
    handle_downed_pokemon,
    handle_level_change,
    handle_xp_change_reward,
)
from skimage.transform import resize
import mediapy as media
from pathlib import Path


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

    def __init__(self, render_mode=None, **options):
        ## CONFIG
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(self.command_map))
        self.instance = random.getrandbits(128)
        ## HANDLE OPTIONS
        for key, value in options.items():
            setattr(self, key, value)

        ## ENV SETTINGS
        self.total_rewards = 0

        ## PYBOY SETTINGS
        window_type = "headless" if render_mode != "human" else "SDL2"
        self.init_state = True
        rom_name = "ROMs/Pokemon Red.gb"
        self.pyboy = PyBoy(
            rom_name,
            window_type=window_type,
        )
        self.pyboy.set_emulation_speed(5)
        self.screen = self.pyboy.botsupport_manager().screen()
        self.screen_shape = (
            72,
            80,
            3,
        )  ## Half the size of the PyBoy screen for better training performance

        ## OBS
        self.long_term_memory_obs = np.zeros(self.screen_shape)
        self.observation_memory = [np.zeros(self.screen_shape) for _ in range(5)]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.screen_shape[0] * 3,
                self.screen_shape[1],
                self.screen_shape[2],
            ),
            dtype=np.uint8,
        )

        ## Game memory
        self.location_memory = np.empty((0, 3))
        self.long_term_memory_dict = {}
        self.prev_long_term_memory_dict = {}
        self.opponent_memory = {}
        self.previous_opponent_memory = {}

    def step(self, action):
        self._send_command(action)
        reward = self._handle_reward()
        self.total_rewards += reward
        self._handle_long_term_memory_observation()
        obs = self._get_obs()
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        options=None,
    ):  # type: ignore
        super().reset(seed=seed, options=options)
        self.seed = seed

        if self.init_state:
            state_name = "ROMs/Pokemon Red.gb.state"
            self.pyboy.load_state(open(state_name, "rb"))
            self.init_state = False

        obs = self._get_obs()
        info = {}
        return obs, info

    def render(self):
        # if self.render_mode == "human":
        #     return None
        obs = self._get_obs()
        return obs

    def close(self):
        self.pyboy.stop()

    def _handle_position_memory(self, reserved_buffer):
        def count_2x2_arrays(rows, columns):
            if rows < 2 or columns < 2:
                return 0
            num_2x2_arrays = (rows // 2) * (columns // 2)
            return num_2x2_arrays

        long_term_position_memory = np.zeros(
            (
                self.screen_shape[0] - reserved_buffer,
                self.screen_shape[1],
                self.screen_shape[2],
            ),
            np.uint8,
        )
        available_space = count_2x2_arrays(
            long_term_position_memory.shape[0], long_term_position_memory.shape[1]
        )
        if available_space <= 0:
            raise Exception("Not enough memory reserved for location history")
        self.location_memory = self.location_memory[-available_space:, :]
        i = 0
        while i < len(self.location_memory):
            for y in range(0, len(long_term_position_memory[0]), 2):
                for x in range(0, len(long_term_position_memory[1]), 2):
                    if i >= len(self.location_memory):
                        break
                    position = self.location_memory[i]
                    position_array = np.array(
                        [[position[0], position[1]], [position[2], 0]]
                    )
                    ## insert the position array into the long term memory
                    long_term_position_memory[y : y + 2, x : x + 2] = position_array[
                        :, :, np.newaxis
                    ]
                    i += 1
        return long_term_position_memory

    def _handle_stats_memory(self):
        self._produce_long_term_memory()
        items_in_long_term_memory = [
            item
            for _, items in self.long_term_memory_dict.values()
            for item in items
            if items is not None
        ]
        ## create the long term memory array
        reserved_buffer = 5
        long_term_stats_memory = np.zeros(
            (reserved_buffer, self.screen_shape[1], self.screen_shape[2]), np.uint8
        )

        i = 0
        while i < len(items_in_long_term_memory):
            for y in range(long_term_stats_memory.shape[0]):
                for x in range(long_term_stats_memory.shape[1]):
                    if i >= len(items_in_long_term_memory):
                        break
                    value = list(items_in_long_term_memory)[i]
                    long_term_stats_memory[y, x, 1] = value
                    i += 1
        return long_term_stats_memory, reserved_buffer

    def _produce_long_term_memory(self):
        """
        Produce the long term memory dictionary
        """
        self.prev_long_term_memory_dict = self.long_term_memory_dict
        location_array = self._get_location_array()
        """
        Each key in the long term memory dict is a tuple of two lists.
        The first list is a list of functions that calculate the reward for that key.
        The second list is a list of the memory addresses that are used to calculate the reward.

        The first list can be None, in which case the reward is not calculated for that key. For these instances, these values are recorded in the long term memory for observation, but not used to calculate the reward.
        """
        self.long_term_memory_dict = {
            "location": (
                None,
                [
                    location_array[0],
                    location_array[1],
                    location_array[2],
                ],
            ),
            "party_hp": (
                [handle_hp_change_reward, handle_downed_pokemon],
                [
                    self.pyboy.get_memory_value(i)
                    for i in [0xD16D, 0xD199, 0xD1C5, 0xD1F1, 0xD21D, 0xD249]
                ],
            ),
            "party_max_hp": (
                None,
                [
                    self.pyboy.get_memory_value(i)
                    for i in [0xD18E, 0xD1BA, 0xD1E6, 0xD212, 0xD23E, 0xD26A]
                ],
            ),
            "party_level": (
                # [handle_level_change],
                None,
                [
                    self.pyboy.get_memory_value(i)
                    for i in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
                ],
            ),
            "pkm_xp": (
                [handle_xp_change_reward],
                [
                    self.pyboy.get_memory_value(0xD17B),
                    self.pyboy.get_memory_value(0xD1A7),
                    self.pyboy.get_memory_value(0xD1D3),
                    self.pyboy.get_memory_value(0xD1FF),
                    self.pyboy.get_memory_value(0xD22B),
                    self.pyboy.get_memory_value(0xD257),
                ],
            ),
            "opponent_pkm_level": (
                None,
                [
                    self.pyboy.get_memory_value(0xCFF3),
                ],
            ),
            "opponent_pkm_hp": (
                None,
                [
                    self.pyboy.get_memory_value(0xCFE7),
                ],
            ),
        }

    def _handle_long_term_memory_observation(self):
        """
        Produce the long term memory observation
        """
        long_term_stats_memory, reserved_buffer = self._handle_stats_memory()
        long_term_position_memory = self._handle_position_memory(reserved_buffer)
        updated_long_term_memory = np.vstack(
            (long_term_stats_memory, long_term_position_memory)
        )
        assert updated_long_term_memory.shape == self.screen_shape
        self.long_term_memory_obs = updated_long_term_memory

    def _handle_reward(self):
        if not self.evaluate_rewards:
            return 0
        new_reward = 0
        new_reward += self._handle_position_reward()
        new_reward += self._handle_long_term_memory_reward()
        return new_reward

    def _get_location_array(self):
        Y = self.pyboy.get_memory_value(0xD361)
        X = self.pyboy.get_memory_value(0xD362)
        M = self.pyboy.get_memory_value(0xD35E)
        location_array = np.array([Y, X, M])
        return location_array

    def _handle_position_reward(self):
        location_array = self._get_location_array()
        # filter self.location_history by M
        map_location_history = self.location_memory[
            np.where(self.location_memory[:, -1] == location_array[-1])
        ]
        # Calculate distances between current location and all previous locations
        distances = np.linalg.norm(map_location_history - location_array, axis=1)
        # Find the minimum distance
        if len(distances) == 0:
            self.location_memory = np.vstack((self.location_memory, location_array))
            if self.verbose:
                print("New location")
            return 1
        min_distance = distances.min()
        if min_distance > 2:
            self.location_memory = np.vstack((self.location_memory, location_array))
            if self.verbose:
                print("New location")
            return 1  # positive reward for moving to a new location
        return 0

    def _handle_long_term_memory_reward(self):
        reward = 0
        for key in self.long_term_memory_dict.keys():
            curr_reward_function, current_memory_value = self.long_term_memory_dict.get(
                key
            )
            if curr_reward_function is None:
                continue
            (
                prev_reward_function,
                previous_memory_value,
            ) = self.prev_long_term_memory_dict.get(key, (None, None))
            if previous_memory_value is None:
                continue
            for func in curr_reward_function:
                reward += func(
                    current_memory_value, previous_memory_value, verbose=self.verbose
                )
        return reward

    def _append_new_observation_to_memory(self):
        ## new screen
        game_pixels_render = self.screen.screen_ndarray()
        ## resize the screens
        game_pixels_render = (
            255 * resize(game_pixels_render, self.screen_shape)
        ).astype(np.uint8)
        ## pop the last observation and add the new one
        self.observation_memory.pop(0)
        self.observation_memory.append(game_pixels_render)
        if len(self.observation_memory) != 5:
            raise Exception("Observation memory is not 5")

    def _create_short_term_memory_stack(self):
        ## Create a 2x2 grid of the last 4 observations
        half_screen_shape = (
            self.screen_shape[0] // 2,
            self.screen_shape[1] // 2,
            self.screen_shape[2],
        )
        previous_observations = [
            (255 * resize(i, half_screen_shape)).astype(np.uint8)
            for i in self.observation_memory[:-1]
        ]
        top_row = np.hstack((previous_observations[3], previous_observations[2]))
        bottom_row = np.hstack((previous_observations[1], previous_observations[0]))
        previous_observations = np.vstack((top_row, bottom_row))
        return previous_observations

    def _get_obs(self):
        self._append_new_observation_to_memory()
        previous_observations = self._create_short_term_memory_stack()
        current_observation = self.observation_memory[-1]
        obs = np.vstack(
            (self.long_term_memory_obs, current_observation, previous_observations)
        )
        return obs

    def _get_info(self):
        return {
            "memory": self.long_term_memory_dict,
            "prev_memory": self.prev_long_term_memory_dict,
            "location_history": self.location_memory,
        }

    def _send_command(self, command) -> np.ndarray:
        """
        Send a command to the emulator and return the new observation. Skip rendering for all but the last frame.
        :param command: The command to send.
        :return: The new observation.
        """
        number_of_frames = 24
        if command in self.command_map:
            sent_release_command = False
            ###
            self.pyboy.send_input(self.command_map[command][0])
            for frame in range(number_of_frames):
                if not sent_release_command and frame >= 8:
                    self.pyboy.send_input(self.command_map[command][1])
                    sent_release_command = True
                if frame == (number_of_frames - 1):  # only render the last frame
                    self.pyboy._rendering(True)
                else:
                    self.pyboy._rendering(False)
                self.pyboy.tick()

    def _extract_screen_chat(self, obs):
        chat_size = 50
        chat = obs[-chat_size:, :, :]
        return chat

    def _get_screen(self, obs):
        image = numpy_array_to_image(obs)
        return image
