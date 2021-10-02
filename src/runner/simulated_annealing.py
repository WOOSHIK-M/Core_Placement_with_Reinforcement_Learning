import math
import random
import numpy as np
import pandas as pd

from typing import Dict, Any, Tuple
from functools import reduce

import utils
from parsedata import DataCollector
from env.reward import RewardCalculator
from structure import Node


def get_switched_vec(x_vec: np.ndarray, y_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Switch two randomly selected nodes."""
    random_order = np.linspace(0, len(x_vec) - 1, len(x_vec), dtype=np.int)
    src = random.choice(random_order)
    while True:
        dst = random.choice(random_order)
        if src != dst: break

    tmp_dst_x, tmp_dst_y = x_vec[dst], y_vec[dst]
    x_vec[dst], y_vec[dst] = x_vec[src], y_vec[src]
    x_vec[src], y_vec[src] = tmp_dst_x, tmp_dst_y

    return x_vec, y_vec


def simulated_annealing(
    calculate_reward: RewardCalculator,
    grid_info: Tuple[int, int, int, int],
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    init_temp_coef: int, 
    n_iters: int,
    gamma: float,
    temp_threshold: float,
) -> float:
    """Do simulated annealing algorithm once."""
    random_order = np.linspace(0, len(x_vec) - 1, len(x_vec), dtype=np.int)
    random.shuffle(random_order)
    x_vec, y_vec = x_vec[random_order], y_vec[random_order]

    # initialize setting
    temp = init_temp_coef * reduce(lambda x, y: x * y, grid_info)
    reward = calculate_reward(x_vec, y_vec)
    best_reward = reward

    while temp > temp_threshold:
        print(f"# temperature: {temp:.4f}, best_reward: {best_reward}")

        for _ in range(n_iters):
            prev_reward = reward

            # switch two nodes
            tmp_x_vec, tmp_y_vec = get_switched_vec(x_vec, y_vec)
            post_reward = calculate_reward(tmp_x_vec, tmp_y_vec)

            delta_e = post_reward - prev_reward
            if delta_e < 0 or np.exp(-delta_e / temp) > random.random():
                x_vec, y_vec = tmp_x_vec, tmp_y_vec
                reward = post_reward
                best_reward = min(reward, best_reward)
        temp *= gamma

    return best_reward


@utils.logging_time
def run(data: DataCollector) -> None:
    """Cores are placed by simulated anneling algorithm."""
    utils.print_title("Simulated Annealing ... ")

    env_config: Dict[str, Any] = data.config["env_config"]
    method_config: Dict[str, Any] = data.config["SA_config"]
    nodes: Dict[int, Node] = data.nodes

    calculate_reward = RewardCalculator(nodes, **env_config["reward_config"])

    reward = simulated_annealing(
        calculate_reward, 
        env_config["grid"], 
        data.phy_core_x, 
        data.phy_core_y, 
        **method_config
    )

    print(
        f"\nBest reward: {reward}\n"
    )
