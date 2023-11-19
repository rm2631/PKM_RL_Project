from datetime import date
import gym
from gym import spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import numpy as np
from skimage.transform import resize
import mediapy as media
import os
import wandb
from datetime import datetime
import random


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
        self.init_state = True

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

        single_screen_size_downscale_ratio = 2
        ### Observation space
        self.single_screen_size = (
            144 // single_screen_size_downscale_ratio,
            160 // single_screen_size_downscale_ratio,
            1,
        )
        self.nb_stacked_screens = 3
        self.stacked_screen_size = (
            self.single_screen_size[0],
            self.single_screen_size[1],
            self.single_screen_size[2] * self.nb_stacked_screens,
        )

        self.position_history_size = 20
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(
                    low=0, high=255, shape=self.stacked_screen_size, dtype=np.uint8
                ),
                "position": spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8),
                "position_history": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.position_history_size, 3),
                    dtype=np.uint8,
                ),
                "party_stats": spaces.Box(
                    ## 6 pokemons times the number of stats
                    low=0,
                    high=255,
                    shape=(6, 6),
                    dtype=np.uint8,
                ),
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

    def _initialize_self(self):
        self.screen_history = [
            np.zeros(self.single_screen_size, dtype=np.uint8)
            for _ in range(self.nb_stacked_screens)
        ]

    def step(self, action):
        self._run_action(action)
        self._update_game_state()
        self._add_video_frame()
        reward = self._handle_reward()
        observation = self._get_obs()
        terminated = False
        truncated = self._get_truncate_status()
        # truncated = False  ##TODO: remove this
        info = self._get_info()
        if truncated or terminated:
            self.video_writer.close()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.seed = seed
        if self.init_state:
            self.init_state = False
            self.reset_seed = random.getrandbits(64)

            reset_attributes = [
                "progress_counter",
                "previous_maps",
                "screen_history",
                # "total_rewards", ##Commented out to keep track of total rewards over time
                "previous_positions",
            ]
            for attribute in reset_attributes:
                if hasattr(self, attribute):
                    delattr(self, attribute)
            state_name = "ROMs/Pokemon Red.gb.state"
            self.pyboy.load_state(open(state_name, "rb"))
            self._initialize_video_writer()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info  # reward, done, info can't be included

    def render(self, mode="human"):
        screen = self._get_screen()
        return screen

    def close(self):
        self.video_writer.close()

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

    def _initialize_video_writer(self):
        if self.configs.get("save_video"):
            save_dir = os.path.join(
                "rollouts", self.configs.get("run_name").name or "run", self.seed
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_name = f"{datetime_str}_{self.reset_seed}.mp4"
            full_path = os.path.join(save_dir, video_name)
            self.video_writer = media.VideoWriter(
                full_path, self.single_screen_size[:2], fps=60
            )
            self.video_writer.__enter__()

    def _add_video_frame(self):
        if self.configs.get("save_video"):
            screen = self._get_screen()[:, :, 0]
            self.video_writer.add_image(screen)

    ## Observation functions

    def _get_obs(self):
        observation = {
            "screen": self._get_screen_stack(),
            "position": self._get_current_position_obs(),
            "position_history": self._get_previous_position_obs(),
            "party_stats": self._get_party_stats_obs(),
        }
        return observation

    def _get_screen(self):
        screen = self.screen.screen_ndarray()
        ## Compress the screen to self.single_screen_size
        screen = resize(screen, self.single_screen_size, anti_aliasing=True)
        return screen

    def _get_previous_screens_array(self):
        screen_stack = np.concatenate(self.screen_history, axis=2)
        return screen_stack

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
        screen_stack_array = self._get_previous_screens_array()

        return screen_stack_array

    def _get_current_position_obs(self):
        if not hasattr(self, "current_position"):
            self.current_position = np.zeros((3), dtype=np.uint8)
        observation = self.current_position
        return observation

    def _get_previous_position_obs(self):
        if not hasattr(self, "previous_positions"):
            self.previous_positions = []
        observation = self.previous_positions[-self.position_history_size :]
        ## Fill with zeros if not enough positions
        if len(observation) < self.position_history_size:
            nb_of_padded_positions = self.position_history_size - len(observation)
            [
                observation.append(np.zeros((3), dtype=np.uint8))
                for _ in range(nb_of_padded_positions)
            ]
        observation = np.array(observation)
        return observation

    def _get_party_stats_obs(self):
        ## stack all party information
        observation = np.array(
            [
                self._get_party_level(),
                self._get_party_hp(),
                self._get_party_max_hp(),
                self._get_party_type(first_type=True),
                self._get_party_type(first_type=False),
                self._get_party_xp(),
            ]
        ).T
        return observation

    def _get_party_level(self):
        party_level = [
            self.pyboy.get_memory_value(i)
            for i in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return party_level

    def _get_party_hp(self):
        party_hp = [
            self.pyboy.get_memory_value(i)
            for i in [0xD16D, 0xD199, 0xD1C5, 0xD1F1, 0xD21D, 0xD249]
        ]
        return party_hp

    def _get_party_max_hp(self):
        party_max_hp = [
            self.pyboy.get_memory_value(i)
            for i in [0xD18E, 0xD1BA, 0xD1E6, 0xD212, 0xD23E, 0xD26A]
        ]
        return party_max_hp

    def _get_party_type(self, first_type=True):
        first_type = [0xD170, 0xD19C, 0xD1C8, 0xD1F4, 0xD220, 0xD24C]
        second_type = [0xD171, 0xD19D, 0xD1C9, 0xD1F5, 0xD221, 0xD24D]
        if first_type:
            type_array = first_type
        else:
            type_array = second_type
        party_type = [self.pyboy.get_memory_value(i) for i in type_array]
        return party_type

    def _get_party_xp(self):
        party_xp = [
            self.pyboy.get_memory_value(i)
            for i in [0xD17B, 0xD1A7, 0xD1D3, 0xD1FF, 0xD22B, 0xD257]
        ]
        return party_xp

    ## Info functions

    def _get_info(self):
        if not hasattr(self, "reward_memory"):
            self.reward_memory = dict()

        info = dict(
            reward_memory=self.reward_memory,
        )
        return info

    ## Reward functions

    def log_reward(weight=1):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                reward = func(self)
                reward = reward * weight
                if reward != 0:
                    self.reward_memory[func.__name__] = reward
                    if self.configs["verbose"]:
                        func_name = func.__name__
                        if func_name not in self.configs["verbose_exclude"]:
                            print(f"---- {func_name}: {reward} ----")
                return reward

            return wrapper

        return decorator

    def _update_game_state(self):
        """
        This function manages the state of the long term memory.
        """
        ##TODO
        pass

    def _update_progress_counter(self, reward):
        if not hasattr(self, "progress_counter"):
            self.progress_counter = 0
        ## Update progress counter
        if reward == 0:
            self.progress_counter += 1
        else:
            self.progress_counter = 0

    def _handle_reward(self):
        ## Reset Rewards Info memory
        self.reward_memory = dict()

        self.step_reward = dict(
            position=self._handle_position_reward(),
            new_map=self._handle_new_map_reward(),
            # xp_gain=self._handle_xp_reward(),
            level_up=self._handle_level_reward(),
            # downed_pokemon=self._handle_downed_pokemon_reward(),
            opponent_hp_loss=self._handle_dealing_dmg_reward(),
            badges=self._handle_badges_reward(),
            healing=self._handle_healing_reward(),
        )

        ## Sum all rewards
        reward = sum(self.step_reward.values())
        if not hasattr(self, "total_rewards"):
            self.total_rewards = 0
        self.total_rewards += reward
        self._update_progress_counter(reward)
        return reward

    @log_reward(weight=0.05)
    def _handle_position_reward(self):
        Y = self.pyboy.get_memory_value(0xD361)
        X = self.pyboy.get_memory_value(0xD362)
        M = self.pyboy.get_memory_value(0xD35E)
        self.current_position = np.array([Y, X, M])

        if not hasattr(self, "previous_positions"):
            self.previous_positions = []

        ## Check if position has changed
        if not any([all(p == self.current_position) for p in self.previous_positions]):
            self.previous_positions.append(self.current_position)
            return min(len(self.previous_positions) * 0.005, 1)
        return 0

    @log_reward(weight=0.2)
    def _handle_new_map_reward(self):
        if not hasattr(self, "previous_maps"):
            self.previous_maps = []
        current_map = self.current_position[2]
        if current_map not in self.previous_maps:
            self.previous_maps.append(current_map)
            return 1
        return 0

    @log_reward(weight=0.5)
    def _handle_xp_reward(self):
        if hasattr(self, "current_party_xp"):
            self.previous_party_xp = self.current_party_xp  ## Update previous party xp
        else:
            self.previous_party_xp = [0 for _ in range(6)]
        self.current_party_xp = self._get_party_xp()
        xp_gain = [
            (current - previous) / previous if previous != 0 else 0
            for current, previous in zip(self.current_party_xp, self.previous_party_xp)
        ]
        ## Reward 1 for level up and xp gain % otherwise
        xp_gain = [min(1, 1 if xp_gain < 0 else xp_gain) for xp_gain in xp_gain]
        reward = sum(xp_gain)
        return reward

    @log_reward(weight=2)
    def _handle_level_reward(self):
        """
        Reward for leveling up or catching a new pokemon.
        """
        party_level = self._get_party_level()
        if not hasattr(self, "max_party_level"):
            self.max_party_level = 6
        current_max_party_level = max(party_level)
        if current_max_party_level <= self.max_party_level:
            return 0
        self.max_party_level = current_max_party_level
        return 1

    @log_reward(weight=0.2)
    def _handle_healing_reward(self):
        """
        Reward for healing the party.
        """
        self.party_hp = self._get_party_hp()
        self.max_party_hp = self._get_party_max_hp()
        if not hasattr(self, "party_hp"):
            self.previous_party_hp = [0 for _ in range(6)]
        else:
            self.previous_party_hp = self.party_hp
        if not hasattr(self, "max_party_hp"):
            self.previous_max_party_hp = [0 for _ in range(6)]
        else:
            self.previous_max_party_hp = self.max_party_hp
        if sum(self.previous_party_hp) == 0:
            ## Cases where the party was blacekd out, we don't want to reward healing
            return 0
        if sum(self.party_hp) == sum(self.max_party_hp) and sum(
            self.previous_party_hp
        ) != sum(self.previous_max_party_hp):
            return 1
        return 0

    @log_reward(weight=0.1)
    def _handle_downed_pokemon_reward(self):
        party_hp_memory_address = [0xD16D, 0xD199, 0xD1C5, 0xD1F1, 0xD21D, 0xD249]
        if hasattr(self, "current_party_hp"):
            self.previous_party_hp = self.current_party_hp
        else:
            self.previous_party_hp = [0 for _ in range(6)]
        self.current_party_hp = [
            self.pyboy.get_memory_value(address) for address in party_hp_memory_address
        ]
        downed_pokemon = [
            True if current == 0 and previous != 0 else False
            for current, previous in zip(self.current_party_hp, self.previous_party_hp)
        ]
        reward = -1 if any(downed_pokemon) else 0
        return reward

    @log_reward(weight=0.01)
    def _handle_dealing_dmg_reward(self):
        """
        Reward for dealing damage to the opponent.
        After the max_level_threshold, this reward is always 0. This is to prevent the agent from grinding only.
        """
        max_level_threshold = self.configs.get("max_level_threshold") or 10
        party_level = self._get_party_level()
        if any([level >= max_level_threshold for level in party_level]):
            return 0
        if not hasattr(self, "current_opp_pkm_hp"):
            self.previous_opp_pkm_hp = 0
        else:
            self.previous_opp_pkm_hp = self.current_opp_pkm_hp
        self.current_opp_pkm_hp = self.pyboy.get_memory_value(0xCFE7)
        if (
            self.current_opp_pkm_hp < self.previous_opp_pkm_hp
            and self.previous_opp_pkm_hp != 0
            and self.current_opp_pkm_hp != 0
        ):
            return 1
        return 0

    @log_reward(weight=5)
    def _handle_badges_reward(self):
        """
        Reward for getting a new badge.
        """
        if not hasattr(self, "current_badge_count"):
            self.previous_badge_count = 0
        else:
            self.previous_badge_count = self.current_badge_count
        badge_memory = self.pyboy.get_memory_value(0xD356)
        self.current_badge_count = bin(badge_memory).count("1")
        if self.current_badge_count > self.previous_badge_count:
            return 1
        return 0

    ## Info functions

    def _get_truncate_status(self):
        if not hasattr(self, "progress_counter"):
            raise ValueError(
                "progress_counter not found. Is this function called too early?"
            )
        max_progress_without_reward = (
            self.configs.get("max_progress_without_reward") or 1000
        )
        tenth_of_max_progress_without_reward = max_progress_without_reward // 10
        if self.progress_counter > 0:
            if self.progress_counter % tenth_of_max_progress_without_reward == 0:
                if self.configs["verbose"]:
                    print(
                        f"Reached {self.progress_counter} of {max_progress_without_reward} steps until truncation"
                    )
        if self.progress_counter >= max_progress_without_reward:
            self.init_state = True
            if self.configs["verbose"]:
                print(f"Truncated at {self.progress_counter} steps")
            return True
        else:
            return False
