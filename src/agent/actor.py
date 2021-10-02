import torch
import torch.nn as nn
import numpy as np

from typing import Dict, Tuple
from torch.distributions import Categorical

from structure import Node
from agent.model import GraphConv
from agent.distributions import DiagGaussian


class Actor(nn.Module):
    def __init__(self, data):
        """Initialize actor."""
        super(Actor, self).__init__()
        env_config = data.config["env_config"]
        rl_config = data.config["RL_config"]

        self.use_cuda = rl_config["use_cuda"]
        self.device = torch.device("cpu")
        if self.use_cuda:
            self.device = torch.device(f"cuda:{rl_config['device']}")

        self.adj_mat = torch.FloatTensor(
            self.get_adjacency_graph(data.lut)
        ).to(self.device)
        self.node_features = torch.FloatTensor(
            self.get_node_features(data.nodes)
        ).to(self.device)

        self.x_range = 1
        self.Y_DIM, self.X_DIM, self.y_dim, self.x_dim = env_config["grid"]
        self.n_outputs = data.n_logic_cores
        """
        GCN layers
        """
        self.gcn = GraphConv(n_features=self.node_features.shape[1], dropout=0.0)

        """
        Gaussian Distribution
        """
        self.dist_x = DiagGaussian(
            self.node_features.shape[0],
            self.node_features.shape[0],
            self.x_range
        )
        self.dist_y = DiagGaussian(
            self.node_features.shape[0],
            self.node_features.shape[0],
            self.x_range
        )

        baseline_x = torch.linspace(-self.x_range, self.x_range, self.x_dim)
        self.x_dist = torch.stack(
            [torch.FloatTensor(baseline_x) for _ in range(self.n_outputs)], 
            dim=1
        ).to(self.device)
        baseline_y = torch.linspace(-self.x_range, self.x_range, self.y_dim)
        self.y_dist = torch.stack(
            [torch.FloatTensor(baseline_y) for _ in range(self.n_outputs)], 
            dim=1
        ).to(self.device)

    def actor_layer(self, actions) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Get actions."""
        gnn_embed = self.gcn(x=self.node_features, adj=self.adj_mat)

        dist_x = self.dist_x(gnn_embed)
        dist_y = self.dist_y(gnn_embed)

        discrete_dist_x = dist_x.log_probs(self.x_dist).exp().T
        discrete_dist_y = dist_y.log_probs(self.y_dist).exp().T

        discrete_dist_x = torch.softmax(discrete_dist_x, dim=1)
        discrete_dist_y = torch.softmax(discrete_dist_y, dim=1)

        if actions is None:
            # sampling
            cate_dist = Categorical(discrete_dist_x)
            action_x = cate_dist.sample()

            cate_dist = Categorical(discrete_dist_y)
            action_y = cate_dist.sample()

            # get log_probs
            action_log_probs_x = discrete_dist_x.gather(dim=1, index=action_x.unsqueeze(1)).squeeze()
            action_log_probs_y = discrete_dist_y.gather(dim=1, index=action_y.unsqueeze(1)).squeeze()

            return (action_x, action_y), (action_log_probs_x, action_log_probs_y)
        else:
            # to update network
            action_x, action_y = actions[0], actions[1]
            action_log_probs_x = torch.empty(action_x.shape).to(self.device)
            for i in range(action_x.shape[0]):
                action_log_probs_x[i] = discrete_dist_x.gather(dim=1, index=action_x[i].unsqueeze(1)).squeeze()

            action_log_probs_y = torch.empty(action_y.shape).to(self.device)
            for i in range(action_y.shape[0]):
                action_log_probs_y[i] = discrete_dist_y.gather(dim=1, index=action_y[i].unsqueeze(1)).squeeze()

            return torch.stack([action_log_probs_x, action_log_probs_y])

    def forward(self, actions=None):
        return self.actor_layer(actions)

    def get_adjacency_graph(self, adj_mat: np.ndarray) -> np.ndarray:
        """Get laplacian normalized adjacency matrix."""
        adj_mat = adj_mat / adj_mat.max()
        
        # Laplacian D^(-1/2) * A * D&(-1/2)
        D = np.zeros(adj_mat.shape)
        for i in range(D.shape[0]):
            D[i, i] = (adj_mat[i] != 0).sum()

        with np.errstate(divide='ignore'):
            D_sqrt = 1.0 / np.sqrt(D)
        D_sqrt = np.where(D_sqrt == np.inf, 0.0, D_sqrt)

        L = D - adj_mat
        L = np.matmul(D_sqrt, L)
        return np.matmul(L, D_sqrt)

    def get_node_features(self, nodes: Dict[int, Node]) -> np.ndarray:
        """Get node features to embeddings."""
        node_features = [list(nodes[i].get_features()) for i in nodes]
        node_features = np.transpose(node_features)
        for idx, features in enumerate(node_features):
            node_features[idx] /= features.max()
        return node_features.T