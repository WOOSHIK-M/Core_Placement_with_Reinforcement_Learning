import math
import numpy as np

from typing import List, Dict

from structure import Node

class RewardCalculator:
    """Reward Calculator which approximates the mapping quality."""

    def __init__(
        self,
        nodes: Dict[int, Node],
        reward_method: str, 
        deadlock_constraint: bool,
        deadlock_coef: float = 0.1
    ) -> None:
        # assign reward calculator
        if reward_method == "Communication_cost":
            self.reward_approximator = CommunicationCost(nodes)
        else:
            NotImplementedError(f"{reward_method} is not supported ...")

        # deadlock penalty
        self.deadlock_constraint = deadlock_constraint
        if self.deadlock_constraint:
            self.deadlock_coef = deadlock_coef
            self.get_deadlock_penalty = DeadlockPenalty(nodes)

    def __call__(self, x_vec: np.ndarray, y_vec: np.ndarray) -> float:
        # reward approximation
        reward = self.reward_approximator(x_vec, y_vec)

        # get deadlock penalty
        if self.deadlock_constraint:
            penalty = self.get_deadlock_penalty(x_vec, y_vec)
            reward *= self.deadlock_coef * math.exp(penalty)
        return reward


class CommunicationCost:
    """Calculate communication cost -> length X weight."""

    def __init__(self, nodes: Dict[int, Node]):
        self.nodes = nodes

    def __call__(self, x_vec: np.ndarray, y_vec: np.ndarray) -> float:
        reward = 0.0
        for sc in self.nodes:
            sx, sy = x_vec[sc], y_vec[sc]

            for dc, packets in self.nodes[sc].to_info.items():
                dx, dy = x_vec[dc], y_vec[dc]

                reward += (math.fabs(sx - dx) + math.fabs(sy - dy)) * packets
                while self.nodes[dc].is_multicast == 1:
                    sx, sy = dx, dy
                    dc = self.nodes[dc].multi_to_core
                    dx, dy = x_vec[dc], y_vec[dc]
                    reward += (math.fabs(sx - dx) + math.fabs(sy - dy)) * packets
        return reward


class DeadlockPenalty:
    """Calculate deadlock constraint."""

    def __init__(self, nodes: Dict[int, Node]):
        self.nodes = nodes
        self.deadlock_connections: int = sum(nodes[idx].is_multicast for idx in self.nodes)

    def __call__(self, x_vec: np.ndarray, y_vec: np.ndarray) -> float:
        deadlock_reward = 0.0
        for i in self.nodes:
            if self.nodes[i].is_multicast:
                mulx, muly = x_vec[i], y_vec[i]

                toc = self.nodes[i].multi_to_core

                to_flag = False
                tox, toy = x_vec[toc], y_vec[toc]
                if tox != mulx:
                    to_flag = True

                from_flag = False
                if to_flag:
                    for sc in self.nodes[i].from_info:
                        fromx, fromy = x_vec[sc], y_vec[sc]
                        if fromy > muly:
                            from_flag = True
                        if from_flag:
                            deadlock_reward += 1
                            from_flag = False
        deadlock_reward /= self.deadlock_connections + 1e-7
        return deadlock_reward