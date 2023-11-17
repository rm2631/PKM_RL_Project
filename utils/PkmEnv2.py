import gym
from gym import spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import numpy as np
from skimage.transform import resize


class PkmEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, **configs):
        super(PkmEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        self.reward_range = (0, 15000)

        ## Env configs
        self.configs = configs
        required_configs = ["rom_path", "render_mode", "emulation_speed"]
        for config in required_configs:
            assert config in configs, f"Missing {config} in configs"

        ### Action space
        self.command_map = {
            1: [WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP],
            2: [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
            3: [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
            4: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
            5: [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
            6: [WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B],
        }
        self.action_space = spaces.Discrete(len(self.command_map))

        ### Observation space
        self.single_screen_size = (
            36,
            40,
            3,
        )
        self.nb_stacked_screens = 3
        self.stacked_screen_size = (
            self.single_screen_size[0] * self.nb_stacked_screens,
            self.single_screen_size[1],
            self.single_screen_size[2],
        )

        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(
                    low=0, high=255, shape=self.stacked_screen_size, dtype=np.uint8
                )
            }
        )

        ## Pyboy
        window_type = "headless" if configs["render_mode"] != "human" else "SDL2"
        self.pyboy = PyBoy(
            configs["rom_path"],
            debugging=False,
            disable_input=False,
            window_type=window_type,
        )
        self.screen = self.pyboy.botsupport_manager().screen()
        self.pyboy.set_emulation_speed(configs["emulation_speed"])

    def step(self, action):
        self._run_action(action)
        observation = self._get_obs()
        reward = self._handle_reward()
        terminated = False
        truncated = self._get_truncate_status()
        # truncated = False  ##TODO: remove this
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.seed = seed

        self.init_state = (
            True  ##TODO: change this value to true as part of the truncation process
        )

        if self.init_state:
            self.init_state = False
            reset_attributes = [
                "progress_counter",
                "screen_history",
                # "total_rewards", ##Commented out to keep track of total rewards over time
                "previous_position",
            ]
            for attribute in reset_attributes:
                if hasattr(self, attribute):
                    delattr(self, attribute)
            state_name = "ROMs/Pokemon Red.gb.state"
            self.pyboy.load_state(open(state_name, "rb"))

        observation = self._get_obs()
        info = self._get_info()
        return observation, info  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    ### Custom methods

    def _run_action(self, action):
        """
        Send a command to the emulator and return the new observation. Skip rendering for all but the last frame.
        :param command: The command to send.
        :return: The new observation.
        """
        number_of_frames = 24
        if action in self.command_map:
            sent_release_command = False
            ###
            self.pyboy.send_input(self.command_map[action][0])
            for frame in range(number_of_frames):
                if not sent_release_command and frame >= 8:
                    self.pyboy.send_input(self.command_map[action][1])
                    sent_release_command = True
                if frame == (number_of_frames - 1):  # only render the last frame
                    self.pyboy._rendering(True)
                else:
                    self.pyboy._rendering(False)
                self.pyboy.tick()

    def _get_obs(self):
        observation = dict(
            screen=self._get_screen_stack(),
        )
        return observation

    def _get_info(self):
        info = dict()
        return info

    def _get_screen(self):
        screen = self.screen.screen_ndarray()
        ## Compress the screen to self.single_screen_size
        screen = resize(screen, self.single_screen_size, anti_aliasing=True)
        return screen

    def _get_screen_stack(self):
        ## if self.screen_history does not exist
        if not hasattr(self, "screen_history"):
            ## create it and fill it with nb_stacked_screens zeros
            self.screen_history = [
                np.zeros(self.single_screen_size, dtype=np.uint8)
                for _ in range(self.nb_stacked_screens)
            ]
        ## add the current screen to the history
        current_screen = self._get_screen()

        ## Add the current screen to the history
        self.screen_history.append(current_screen)

        ## Keep only the last nb_stacked_screens screens
        self.screen_history = self.screen_history[-self.nb_stacked_screens :]

        ## Stack the screens
        screen_stack = np.concatenate(self.screen_history, axis=0)

        return screen_stack

    ## Reward functions

    def print_reward(func):
        def wrapper(self):
            reward = func(self)
            if reward != 0 and self.configs["verbose"]:
                func_name = func.__name__
                print(f"---- {func_name}: {reward} ----")
            return reward

        return wrapper

    def _update_game_state(self):
        pass

    def _update_progress_counter(self, reward):
        if not hasattr(self, "progress_counter"):
            self.progress_counter = 0
        ## Update progress counter
        if reward == 0:
            self.progress_counter += 1
        else:
            self.progress_counter = 0
        print(f"Progress counter: {self.progress_counter}")  ##TODO: remove this

    def _handle_reward(self):
        self.step_reward = dict(
            position=self._handle_position_reward() * 0.1,
        )

        ## Sum all rewards
        reward = sum(self.step_reward.values())
        if not hasattr(self, "total_rewards"):
            self.total_rewards = 0
        self.total_rewards += reward
        self._update_progress_counter(reward)
        return reward

    @print_reward
    def _handle_position_reward(self):
        Y = self.pyboy.get_memory_value(0xD361)
        X = self.pyboy.get_memory_value(0xD362)
        M = self.pyboy.get_memory_value(0xD35E)
        current_position = np.array([Y, X, M])
        if not hasattr(self, "previous_position"):
            self.previous_position = []
        self.previous_position.append(current_position)
        ## Calculate reward
        if len(self.previous_position) <= 1:
            return 1
        ## Filter positions that share same last coordinate
        filtered_positions = [
            pos for pos in self.previous_position if pos[2] == current_position[2]
        ]
        ## Calculate distance between last two positions
        distances = np.linalg.norm(filtered_positions - current_position, axis=1)
        if distances.min() > 2:
            return 1
        ## If all else fails, return 0
        return 0

    ## Info functions

    def _get_truncate_status(self):
        if not hasattr(self, "progress_counter"):
            raise ValueError(
                "progress_counter not found. Is this function called too early?"
            )
        max_progress_without_reward = (
            self.configs.get("max_progress_without_reward") or 10000
        )
        if self.progress_counter >= max_progress_without_reward:
            self.init_state = True
            return True
        else:
            return False
