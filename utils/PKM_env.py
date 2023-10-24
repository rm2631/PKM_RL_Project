
import sys
import uuid 
import os
from math import floor, sqrt
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pyboy import PyBoy
import pandas as pd
from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from utils.PKM_PyBoy import PKM_PyBoy



class PKM_env(Env):
    def __init__(self):
        self.pyboy = PKM_PyBoy()

    def step(
        self, action
    ) :
        pass

    def reset(
        self,
        *,
        seed,
        options,
    ):  # type: ignore
        pass
        super().reset(seed=seed, options=options)

    def render(self):
        pass

    def close(self):
        pass