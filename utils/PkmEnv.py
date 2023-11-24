from datetime import date
from hmac import new
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
import json


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
        self.map_history_size = 15
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
                "map_history": spaces.Box(
                    low=0, high=255, shape=(self.map_history_size,), dtype=np.uint8
                ),
                "party_stats": spaces.Box(
                    ## 6 pokemons times the number of stats
                    low=0,
                    high=255,
                    shape=(6, 6),
                    dtype=np.uint8,
                ),
                "step_progression": spaces.Box(
                    low=0, high=255, shape=(2,), dtype=np.uint8
                ),
                ##TODO: Add to obs : badges, opponent stats
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

        ## Game State
        self._initialize_self()

    def _initialize_self(self):
        self.screen_history = [
            np.zeros(self.single_screen_size, dtype=np.uint8)
            for _ in range(self.nb_stacked_screens)
        ]
        self.current_position = np.zeros((3), dtype=np.uint8)
        self.previous_positions = []
        self.previous_rewarded_positions = []
        self.reward_memory = dict()
        self.progress_counter = 0
        self.total_rewards = 0
        self.previous_relevant_maps = []
        self.max_party_level = 6
        ## HP
        self.previous_party_hp = self._get_party_hp()
        self.current_party_hp = self._get_party_hp()
        self.current_max_party_hp = self._get_party_max_hp()
        self.previous_max_party_hp = self._get_party_max_hp()
        ## OPP HP
        self.previous_opp_pkm_hp = 0
        self.current_opp_pkm_hp = 0
        ## Badges
        self.current_badge_count = 0
        self.previous_badge_count = 0
        ## XP
        self.current_party_xp = self._get_party_xp()
        self.previous_party_xp = self._get_party_xp()

    def _log_obs(self):
        try:
            max_progress_without_reward = self.configs.get(
                "max_progress_without_reward"
            )
            tenth_of_max_progress_without_reward = max_progress_without_reward // 10
            if self.progress_counter % tenth_of_max_progress_without_reward == 0:
                obs = self._get_obs()
                ##pop the screen
                obs.pop("screen")
                for key, value in obs.items():
                    obs[key] = value.tolist()
                path = self._get_path("logs")
                file_name = self._get_file_name(".json")
                ## SAVE TO JSON
                full_path = os.path.join(path, file_name)
                with open(full_path, "w") as f:
                    json.dump(obs, f)
        except:
            pass

    def step(self, action):
        self._run_action(action)
        self._update_game_state()
        self._add_video_frame()
        reward = self._handle_reward()
        observation = self._get_obs()
        terminated = False
        truncated = self._get_truncate_status()
        info = self._get_info()
        if truncated or terminated:
            self.video_writer.close()
        self._log_obs()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.seed = seed
        if self.init_state:
            self.init_state = False
            self.reset_seed = random.getrandbits(64)
            self._initialize_self()
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
        wandb.finish()

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

    def _get_path(self, prefix):
        path = os.path.join(
            "run_logs",
            self.configs.get("run_id"),
            prefix,
            self.configs.get("run_name").name,
        )
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _get_file_name(self, extension):
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{datetime_str}_{self.reset_seed}" + extension
        return file_name

    def _initialize_video_writer(self):
        if self.configs.get("save_video"):
            path = self._get_path("rollouts")
            file_name = self._get_file_name(".mp4")
            full_path = os.path.join(path, file_name)
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
            "map_history": self._get_map_history_obs(),
            "party_stats": self._get_party_stats_obs(),
            "step_progression": np.array(
                [self.progress_counter, self.configs.get("max_progress_without_reward")]
            ),
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
        observation = self.current_position
        return observation

    def _get_previous_position_obs(self):
        observation = self.previous_positions[-self.position_history_size :]
        observation.reverse()
        ## Fill with zeros if not enough positions
        if len(observation) < self.position_history_size:
            nb_of_padded_positions = self.position_history_size - len(observation)
            [
                observation.append(np.zeros((3), dtype=np.uint8))
                for _ in range(nb_of_padded_positions)
            ]
        observation = np.array(observation)
        return observation

    def _get_map_history_obs(self):
        ## Get list of unique maps from previous positions
        previous_maps = list(set([p[2] for p in self.previous_positions]))
        observation = previous_maps[-self.map_history_size :]
        observation.reverse()
        ## Fill with zeros if not enough positions
        if len(observation) < self.map_history_size:
            nb_of_padded_positions = self.map_history_size - len(observation)
            [observation.append(0) for _ in range(nb_of_padded_positions)]
        observation = np.array(observation).T
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
                    func_name = func.__name__
                    self.reward_memory[func_name] = reward
                    ## Update total reward
                    func_total_name = f"total_{func_name}"
                    if not hasattr(self, func_total_name):
                        setattr(self, func_total_name, 0)
                    total_reward = getattr(self, func_total_name)
                    new_reward = total_reward + reward
                    setattr(self, func_total_name, new_reward)
                    if self.configs["verbose"]:
                        if func_name not in self.configs["verbose_exclude"]:
                            print(f"---- {func_name}: {reward} ----")
                    wandb.log({func_total_name: total_reward})
                return reward

            return wrapper

        return decorator

    def _update_game_state(self):
        """
        This function manages the state of the long term memory.
        """
        self.current_position = self._get_position()
        self.previous_positions.append(self.current_position)

    def _get_position(self):
        Y = self.pyboy.get_memory_value(0xD361)
        X = self.pyboy.get_memory_value(0xD362)
        M = self.pyboy.get_memory_value(0xD35E)
        current_position = np.array([Y, X, M])
        return current_position

    def _update_progress_counter(self, reward):
        ## Update progress counter
        if reward <= 0:
            self.progress_counter += 1
        else:
            self.progress_counter = 0

    def _handle_reward(self):
        ## Reset Rewards Info memory
        self.reward_memory = dict()

        self.step_reward = dict(
            position=self._handle_position_reward(),
            relevant_location=self._handle_relevant_location_reward(),
            level_up=self._handle_level_reward(),
            downed_pokemon=self._handle_downed_pokemon_reward(),
            opponent_hp_loss=self._handle_dealing_dmg_reward(),
            badges=self._handle_badges_reward(),
            healing=self._handle_healing_reward(),
            # max_steps_without_reward=self._handle_max_steps_without_reward(),
        )

        ## Sum all rewards
        reward = sum(self.step_reward.values())
        self._update_total_rewards(reward)
        self._update_progress_counter(reward)
        return reward

    def _update_total_rewards(self, reward):
        self.total_rewards += reward
        wandb.log({"total_rewards": self.total_rewards})

    @log_reward(weight=0.01)
    def _handle_position_reward(self):
        ##TODO: Faire la différence entre les previous locations et les rewarded locations
        ##TODO: Ajouter les cartes dans les observations

        def filter_maps(position, history):
            current_map = position[2]
            filtered_maps = [p for p in history if p[2] == current_map]
            return filtered_maps

        def get_min_distance(position, history):
            if len(history) == 0:
                self.previous_rewarded_positions.append(self.current_position)
                return 0
            distances = [np.linalg.norm(position - p) for p in history]
            return min(distances)

        previous_rewarded_positions = filter_maps(
            self.current_position, self.previous_rewarded_positions
        )
        min_distance = get_min_distance(
            self.current_position, previous_rewarded_positions
        )
        if min_distance > 1:
            self.previous_rewarded_positions.append(self.current_position)
            return min(len(self.previous_rewarded_positions), 100)
        return 0

    @log_reward(weight=3)
    def _handle_relevant_location_reward(self):
        current_map = self.current_position[2]
        if current_map in self.relevant_game_locations:
            if current_map not in self.previous_relevant_maps:
                self.previous_relevant_maps.append(current_map)
                return 1
        return 0

    @log_reward(weight=2)
    def _handle_level_reward(self):
        """
        Reward for leveling up or catching a new pokemon.
        """
        party_level = self._get_party_level()
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
        ## Set previous values
        self.previous_party_hp = self.current_party_hp
        self.previous_max_party_hp = self.current_max_party_hp
        ## Set current values
        self.current_party_hp = self._get_party_hp()
        self.current_max_party_hp = self._get_party_max_hp()

        if sum(self.previous_party_hp) == 0:
            ## Cases where the party was blacked out, we don't want to reward healing
            return 0
        if sum(self.previous_party_hp) == sum(self.previous_max_party_hp):
            ## Cases where the party was already full, we don't want to reward healing
            return 0
        if sum(self.current_party_hp) == sum(self.current_max_party_hp):
            ## Cases where the party is full, we want to reward healing
            return 1
        return 0

    @log_reward(weight=-0.1)
    def _handle_downed_pokemon_reward(self):
        party_hp_memory_address = [0xD16D, 0xD199, 0xD1C5, 0xD1F1, 0xD21D, 0xD249]
        self.previous_party_hp = self.current_party_hp
        self.current_party_hp = [
            self.pyboy.get_memory_value(address) for address in party_hp_memory_address
        ]
        downed_pokemon = [
            True if current == 0 and previous != 0 else False
            for current, previous in zip(self.current_party_hp, self.previous_party_hp)
        ]
        reward = 1 if any(downed_pokemon) else 0
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
        self.previous_badge_count = self.current_badge_count
        badge_memory = self.pyboy.get_memory_value(0xD356)
        self.current_badge_count = bin(badge_memory).count("1")
        if self.current_badge_count > self.previous_badge_count:
            return 1
        return 0

    @log_reward(weight=-0.3)
    def _handle_max_steps_without_reward(self):
        """
        Reward for reaching max_steps_without_reward.
        """
        max_progress_without_reward = self.configs.get("max_progress_without_reward")
        if (self.progress_counter + 1) >= max_progress_without_reward:
            return 1
        return 0

    ## Info functions

    def _get_truncate_status(self):
        max_progress_without_reward = self.configs.get("max_progress_without_reward")
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
