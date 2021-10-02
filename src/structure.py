from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import json


@dataclass
class Node:
    """Initialize node structure."""
    
    # send packets (keys: to_where, values: how_many)
    to_info: Dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # # receive packets (keys: from_where, values: how_many)
    from_info: Dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # multicast info
    is_multicast: bool = False
    multi_to_core: Optional[int] = None
    multi_from_core: Optional[int] = None

    def get_features(self) -> Tuple[int, float, float, float, float]:
        """Get node features.
        
        Features:
            is_multicast, in_degree, out_degree, in_weight, out_weight
        """
        is_multicast = int(self.is_multicast)
        in_degree = len(self.from_info.keys())
        out_degree = len(self.to_info.keys())
        in_weight = sum(self.from_info.values()) / max(in_degree, 1)
        out_weight = sum(self.to_info.values()) / max(out_degree, 1)

        return (
            is_multicast,
            in_degree,
            out_degree,
            in_weight,
            out_weight
        )

    def print_info(self, idx: int = 0):
        """Print out node information."""
        print(
            f"node_idx: {idx}".ljust(15),
            
            f"| connected_to_cores: {list(self.to_info.keys())}".ljust(30),
            f"| connected_to_packets: {list(self.to_info.values())}".ljust(30),

            f"| connected_from_cores: {list(self.from_info.keys())}".ljust(30),
            f"| connected_from_packets: {list(self.from_info.values())}".ljust(30),

            f"| is_multicast: {self.is_multicast} ({self.multi_to_core})".ljust(30),
            "|\n"
        )

    def __hash__(self) -> int:
        """Use instance id as a hash value."""
        return id(self)