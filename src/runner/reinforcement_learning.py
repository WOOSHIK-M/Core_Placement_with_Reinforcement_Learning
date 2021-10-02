import math
import random
import numpy as np
import pandas as pd

from typing import Dict, Any, Tuple
from functools import reduce

import utils
from parsedata import DataCollector
from agent.agents import PPOAgent


@utils.logging_time
def run(data: DataCollector) -> None:
    """Cores are placed by simulated anneling algorithm."""
    utils.print_title("Reinforcement Learning ... ")

    agent = PPOAgent(data)
    agent.train()

    

    


