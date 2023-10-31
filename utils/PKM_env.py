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
from utils.numpy_array_to_image_and_save import numpy_array_to_image_and_save

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
        self.pyboy.set_emulation_speed(5)
        self.instance = random.getrandbits(128)
        self.screen = self.pyboy.botsupport_manager().screen()
        self.location_history = np.empty((0, 3))
        self.action_space = spaces.Discrete(len(self.command_map))
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=int)
        self.evaluate_rewards = kwargs.get("evaluate_rewards", False)
        self.last_party_hp = 0
        self.last_opponent_party_hp = 0

    def step(
        self, action
    ) :
        obs = self._send_command(action)
        reward = self._handle_reward(obs)
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
        new_reward += self._handle_position(obs)
        new_reward += self._handle_party_hp(obs)
        new_reward += self._handle_opponent_party_hp(obs)

        if new_reward > 0:
            print(f"Reward: {new_reward}")
            self._save_screen()
        return new_reward
    

    
    def _handle_position(self, obs):
        Y = self.pyboy.get_memory_value(0xD361)
        X = self.pyboy.get_memory_value(0xD362)
        M = self.pyboy.get_memory_value(0xD35E)
        location_array = np.array([Y, X, M])
        if self.location_history is None:
            self.location_history = np.vstack((self.location_history, location_array))
            return 1
        #filter self.location_history by M
        map_location_history = self.location_history[np.where(self.location_history[:, -1] == M)]
        if len(map_location_history) == 0:
            self.location_history = np.vstack((self.location_history, location_array))
            return 1
        #Calculate distances between current location and all previous locations
        distances = np.linalg.norm(map_location_history - location_array, axis=1)
        #Find the minimum distance
        min_distance = distances.min()
        if min_distance > 1:
            self.location_history = np.vstack((self.location_history, location_array))
            return 1 # positive reward for moving to a new location
        return 0
    
    def _handle_party_hp(self, obs):
        current_party_hp = 0
        hp_memory_addresses = [0xD16D, 0xD199, 0xD1C5, 0xD1F1, 0xD21D, 0xD249]
        for address in hp_memory_addresses:
            current_party_hp += self.pyboy.get_memory_value(address)    
        if current_party_hp > self.last_party_hp:
            self.last_party_hp = current_party_hp
            return 2 # positive reward for increasing total hp, either by healing or leveling up
        return 0
    
    def _handle_opponent_party_hp(self, obs):
        current_opponent_party_hp = 0
        hp_memory_addresses = [0xD8A6, 0xD8D2, 0xD8FE, 0xD92A, 0xD956, 0xD982]
        for address in hp_memory_addresses:
            current_opponent_party_hp += self.pyboy.get_memory_value(address)
        if current_opponent_party_hp < self.last_opponent_party_hp:
            self.last_opponent_party_hp = current_opponent_party_hp
            return 1 # positive reward for decreasing opponent's total hp
        return 0
    
    def _get_obs(self):
        game_pixels_render = self.screen.screen_ndarray()
        return game_pixels_render
    
    def _get_info(self):
        return {}
    
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
                if frame == (number_of_frames - 1): # only render the last frame
                    self.pyboy._rendering(True)
                else:
                    self.pyboy._rendering(False)
                self.pyboy.tick()
        obs = self._get_obs()
        return obs
    
    def _extract_screen_chat(self, obs):
        chat_size = 50
        chat = obs[-chat_size:, :, :]
        return chat

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

