import math
import random
import numpy as np
import pandas as pd

from typing import Dict, Any
from functools import reduce

import utils
from parsedata import DataCollector
from env.reward import RewardCalculator
from structure import Node


def get_random_reward(calculate_reward: RewardCalculator, x_vec: np.ndarray, y_vec: np.ndarray) -> float:
    """Random search implementation."""
    random_order = np.linspace(0, len(x_vec) - 1, len(x_vec), dtype=np.int)
    random.shuffle(random_order)

    return calculate_reward(x_vec[random_order], y_vec[random_order])
    


@utils.logging_time
def run(data: DataCollector) -> None:
    """Cores are placed randomly."""
    utils.print_title("Random Search ... ")

    env_config: Dict[str, Any] = data.config["env_config"]
    method_config: Dict[str, Any] = data.config["RS_config"]
    nodes: Dict[int, Node] = data.nodes

    calculate_reward = RewardCalculator(nodes, **env_config["reward_config"])

    best_reward = 1e10
    for _ in range(method_config["repeat_num"]):
        tmp_reward = get_random_reward(calculate_reward, data.phy_core_x, data.phy_core_y)
        best_reward = min(tmp_reward, best_reward)

    print(
        f"Best reward: {best_reward}\n"
    )
