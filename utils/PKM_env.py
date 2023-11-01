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
from skimage.transform import resize

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
        ## CONFIG
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(self.command_map))
        self.instance = random.getrandbits(128)
        self.evaluate_rewards = kwargs.get("evaluate_rewards", False)




        ## PYBOY
        window_type = 'headless' if render_mode != 'human' else 'SDL2'
        self.pyboy = PyBoy(
            'ROMs/Pokemon Red.gb',
            window_type=window_type,
        )
        self.pyboy.set_emulation_speed(5)
        self.screen = self.pyboy.botsupport_manager().screen()
        self.screen_shape = (72, 80, 3) ## Half the size of the PyBoy screen

        ## OBS
        self.long_term_memory_obs = np.empty(self.screen_shape)
        self.observation_memory = [np.empty(self.screen_shape) for _ in range(5)]
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.screen_shape[0] * 3 , self.screen_shape[1], self.screen_shape[2]), 
            dtype=np.uint8
        )

        ## Game memory
        self.location_memory = np.empty((0, 3))
        self.long_term_memory_dict = {}
        self.prev_long_term_memory_dict = {}

    def step(
        self, action
    ) :
        self._send_command(action)
        self._handle_long_term_memory()
        reward = self._handle_reward()
        obs = self._get_obs()
        if reward > 0:
            self._save_screen(obs)
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

    def _handle_long_term_memory(self):
        self.prev_long_term_memory_dict = self.long_term_memory_dict

        pkm_1_max_hp = self.pyboy.get_memory_value(0xD18E)
        pkm_1_current_hp = self.pyboy.get_memory_value(0xD16D)
        pkm_1_xp = self.pyboy.get_memory_value(0xD17B)
        pkm_1_level = self.pyboy.get_memory_value(0xD18C)

        self.long_term_memory_dict = {
            "pkm_1_max_hp": pkm_1_max_hp,
            "pkm_1_current_hp": pkm_1_current_hp,
            "pkm_1_xp": pkm_1_xp,
            "pkm_1_level": pkm_1_level,
        }

        updated_long_term_memory = np.zeros(self.screen_shape)
        for i, (key, value) in enumerate(self.long_term_memory_dict.items()):
            if i >= len(self.long_term_memory_obs):
                break
            updated_long_term_memory[i, 0, 0] = int(value)
        self.long_term_memory_obs = updated_long_term_memory

    def _handle_reward(self):
        if not self.evaluate_rewards:
            return 0
        new_reward = 0
        new_reward += self._handle_position_reward()
        new_reward += self._handle_memory_reward()
        return new_reward
    
    def _handle_memory_reward(self):
        reward = 0
        for key in self.long_term_memory_dict.keys():
            previous_memory_value = self.prev_long_term_memory_dict.get(key, None)
            current_memory_value = self.long_term_memory_dict.get(key, None)
            if previous_memory_value is None or current_memory_value is None:
                continue
            ## this works because the game benefits from increasing all evaluated values.
            if current_memory_value > previous_memory_value:
                reward += 5
        return reward
    
    def _handle_position_reward(self):
        Y = self.pyboy.get_memory_value(0xD361)
        X = self.pyboy.get_memory_value(0xD362)
        M = self.pyboy.get_memory_value(0xD35E)
        location_array = np.array([Y, X, M])
        
        if self.location_memory is None:
            self.location_memory = np.vstack((self.location_memory, location_array))
            return 1
        #filter self.location_history by M
        map_location_history = self.location_memory[np.where(self.location_memory[:, -1] == M)]
        if len(map_location_history) == 0:
            self.location_memory = np.vstack((self.location_memory, location_array))
            return 1
        #Calculate distances between current location and all previous locations
        distances = np.linalg.norm(map_location_history - location_array, axis=1)
        #Find the minimum distance
        min_distance = distances.min()
        if min_distance > 1:
            self.location_memory = np.vstack((self.location_memory, location_array))
            return 1 # positive reward for moving to a new location
        return 0
    
    def _append_new_observation_to_memory(self):
        ## new screen
        game_pixels_render = self.screen.screen_ndarray()
        ## resize the screens
        game_pixels_render = (255*resize(game_pixels_render, self.screen_shape)).astype(np.uint8)
        ## pop the last observation and add the new one
        self.observation_memory.pop(0)
        self.observation_memory.append(game_pixels_render)
        if len(self.observation_memory) != 5:
            raise Exception("Observation memory is not 5")
        
    def _create_short_term_memory_stack(self):     
        ## Create a 2x2 grid of the last 4 observations
        half_screen_shape = (self.screen_shape[0] // 2, self.screen_shape[1] // 2, self.screen_shape[2])
        previous_observations = [
                (255*resize(i, half_screen_shape)).astype(np.uint8)
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
        obs = np.vstack((self.long_term_memory_obs, current_observation, previous_observations))
        return obs
    
    def _get_info(self):
        return {"memory": self.long_term_memory_dict, "prev_memory": self.prev_long_term_memory_dict, "location_history": self.location_memory}
    
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
    
    def _extract_screen_chat(self, obs):
        chat_size = 50
        chat = obs[-chat_size:, :, :]
        return chat

    def _save_screen(self, obs):
        now = datetime.now()
        day_string = now.strftime("%d-%m-%Y")
        directory = f"screenshots/{day_string}/{self.instance}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        numpy_array_to_image_and_save(obs, f"{directory}")
