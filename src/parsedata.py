import os
import math
import numpy as np

from typing import Tuple, Dict, Any
from collections import defaultdict
from runpy import run_path
from functools import reduce

from structure import Node
from utils import logging_time
from data.toy_lut import get_toy_lut


class DataCollector:
    def __init__(self, config_path: str):
        self.config: Dict[str, Any] = self.get_config(config_path)

        self.n_logic_cores, self.lut, self.is_multicast = self.get_lut_info()
        self.nodes: Dict[int, Node] = self.get_nodes()
        self.phy_core_x, self.phy_core_y, self.phy_chip_x, self.phy_chip_y  = self.get_base_coor()

    @logging_time
    def get_config(self, config_path: str) -> Dict[str, Any]:
        """Initialize the package."""
        assert os.path.exists(config_path), "Config file does not exist ..."

        # get configs
        config = run_path(config_path)["config"]
    
        return config

    @logging_time
    def get_lut_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """Load netlist data."""
        filename = self.config["lut_info"]["lut_file"]
        filepath = os.path.join("data", filename)

        # get lut data
        if filename == "toy_lut.txt":
            get_toy_lut()
        else:
            # load lut file
            if not os.path.exists(filepath):
                NotImplementedError(f"{filepath} does not exist ...")
        lut_info = np.loadtxt(filepath)
        lut_info = np.array(lut_info, dtype=np.int)

        # get basic info of lut
        n_logic_cores = lut_info.shape[1]

        assert n_logic_cores <= reduce(lambda x, y: x * y, self.config["env_config"]["grid"]), "Canvas too small !"

        lut = lut_info[:n_logic_cores]
        is_multicast = lut_info[n_logic_cores:]

        return n_logic_cores, lut, is_multicast

    @logging_time
    def get_nodes(self) -> Dict[int, Node]:
        """Get logic cores data."""
        nodes = {idx: Node() for idx in range(self.n_logic_cores)}

        # normal connections info
        for src, to_packets in enumerate(self.lut):
            to_where = np.nonzero(to_packets)

            # store connection info
            for dst in to_where[0]:
                nodes[src].to_info[dst] = to_packets[dst]
                nodes[dst].from_info[src] = to_packets[dst]

        # multicast connections info
        if self.config["lut_info"]["is_multicast"]:
            for src, to_packets in enumerate(self.is_multicast):
                multi_to_core = np.nonzero(to_packets)[0]

                if len(multi_to_core) > 0:
                    nodes[src].is_multicast = True
                    nodes[src].multi_to_core = multi_to_core[0]
                    nodes[multi_to_core[0]].multi_from_core = src
        return nodes

    @logging_time
    def get_base_coor(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate x, y coordinates according to zigzag"""
        row_chip, col_chip, row_core, col_core = self.config["env_config"]["grid"]

        n_phy_chips = row_chip * col_chip
        n_phy_nodes_per_chips = row_core * col_core
        n_phy_nodes = n_phy_chips * n_phy_nodes_per_chips
        
        phy_core_x, phy_core_y, phy_chip_x, phy_chip_y = [], [], [], []
        for i in range(n_phy_nodes):
            core_idx = i
            chip_idx = 0

            while core_idx >= n_phy_nodes_per_chips:
                core_idx -= n_phy_nodes_per_chips
                chip_idx += 1

            chip_x = chip_idx % col_chip
            chip_y = math.floor(chip_idx / col_chip)

            phy_chip_x.append(chip_x)
            phy_chip_y.append(chip_y)
            phy_core_x.append(core_idx % col_core + chip_x * col_core)
            phy_core_y.append(math.floor(core_idx / col_core) + chip_y * row_core)

        return np.array(phy_core_x), np.array(phy_core_y), np.array(phy_chip_x), np.array(phy_chip_y) 
        