import numpy as np

from typing import Dict, Any
from functools import reduce

import utils
from parsedata import DataCollector
from env.reward import RewardCalculator
from structure import Node


def zigzag(calculate_reward: RewardCalculator, x_vec: np.ndarray, y_vec: np.ndarray) -> float:
    """Zigzag implementation."""
    return calculate_reward(x_vec, y_vec)

def neighbor(calculate_reward: RewardCalculator, x_vec: np.ndarray, y_vec: np.ndarray, n_cols: int) -> float:
    """Neighbor implementation."""
    x_vec = np.where(np.array(y_vec) % 2 == 0, np.array(x_vec), n_cols - np.array(x_vec) - 1)
    return calculate_reward(x_vec, y_vec)


@utils.logging_time
def run(data: DataCollector) -> None:
    """Cores are placed by Zigzag and Neighbor."""
    utils.print_title("Exact Mapping ... ")

    env_config: Dict[str, Any] = data.config["env_config"]
    nodes: Dict[int, Node] = data.nodes

    calculate_reward = RewardCalculator(nodes, **env_config["reward_config"])

    zigzag_reward = zigzag(calculate_reward, data.phy_core_x, data.phy_core_y)
    neighbor_reward = neighbor(calculate_reward, data.phy_core_x, data.phy_core_y, env_config["grid"][-1])

    print(f"Zigzag mapping: {zigzag_reward}")
    print(f"Neighbor mapping: {neighbor_reward}")
    print()
