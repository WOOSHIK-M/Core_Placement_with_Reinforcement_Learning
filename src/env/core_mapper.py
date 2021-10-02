from random import randrange
import torch

import numpy as np

from parsedata import DataCollector
from env.reward import RewardCalculator


class CoreMapper:
    def __init__(self, data: DataCollector):
        """Initialize environment."""
        rl_config = data.config["RL_config"]

        self.use_cuda = rl_config["use_cuda"]
        self.device = torch.device("cpu")
        if self.use_cuda:
            self.device = torch.device(f"cuda:{rl_config['device']}")

        self.n_logic_cores = data.n_logic_cores
        self.env_config = data.config["env_config"]

        self.Y_DIM, self.X_DIM, self.y_dim, self.x_dim = self.env_config["grid"]
        self.n_chips = self.X_DIM * self.Y_DIM
        self.n_nodes_per_chip = self.x_dim * self.y_dim
        self.n_nodes = self.n_chips * self.n_nodes_per_chip

        self.calculate_reward = RewardCalculator(data.nodes, **self.env_config["reward_config"])

    def step(self, actions: torch.Tensor):
        """Single step execution."""
        if self.use_cuda:
            actions = actions.cpu()
        
        # row, col
        actions= np.stack([actions[1].numpy(), actions[0].numpy()]).T
        
        x_vec = [-1 for _ in range(self.n_logic_cores)]
        y_vec = [-1 for _ in range(self.n_logic_cores)]

        overlapped = [False for _ in range(self.n_logic_cores)]
        placement_map = np.zeros((self.y_dim, self.x_dim))

        # mapping cores in order
        for idx, action in enumerate(actions):
            row, col = action

            if placement_map[row, col] == 0:
                y_vec[idx], x_vec[idx] = row, col
                placement_map[row, col] = idx + 1
            else:
                overlapped[idx] = True

        # mapping overlapped cores
        for idx, is_overlap in enumerate(overlapped):
            if is_overlap:
                distance = 1
                tempy, tempx = actions[idx][0], actions[idx][1]

                while True:
                    tempy -= 1
                    tx, ty = self.detect_circle(distance, tempx, tempy, placement_map)
                    if tx != -1:
                        tempx, tempy = tx, ty
                        break
                    else:
                        distance += 1
                placement_map[tempy, tempx] = idx + 1
                x_vec[idx], y_vec[idx] = tempx, tempy
        
        return -self.calculate_reward(x_vec, y_vec)                

    def detect_circle(self, distance, tempx, tempy, placement_map):
        """Detect a vacant node in a round."""
        # Go right & down
        for _ in range(distance):
            tempx += 1
            tempy += 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if placement_map[tempy][tempx] == 0:
                    return tempx, tempy

        # Go left & down
        for _ in range(distance):
            tempx -= 1
            tempy += 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if placement_map[tempy][tempx] == 0:
                    return tempx, tempy

        # Go left & up
        for _ in range(distance):
            tempx -= 1
            tempy -= 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if placement_map[tempy][tempx] == 0:
                    return tempx, tempy

        # Go right & up
        for _ in range(distance):
            tempx += 1
            tempy -= 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if placement_map[tempy][tempx] == 0:
                    return tempx, tempy

        return -1, -1