import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import time

from Code.Distributions import DiagGaussian, DiagBeta
from Code.Layers import GraphConvolution


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()

        self.use_cuda = config.use_cuda
        self.device = config.device

        self.mode = config.params_dict['mode']

        self.lutnum = config.params_dict['lutnum']

        self.x_dim = config.params_dict['x_dim']
        self.y_dim = config.params_dict['y_dim']
        self.allnodenum = config.params_dict['allnodenum']
        self.NodePerChip = self.x_dim * self.y_dim

        self.feature_num = config.params_dict['feature_num']
        self.readout_num = config.params_dict['readout_num']
        self.batch_size = config.params_dict['batch_size']

        self.x_range = config.params_dict['x_range']
        self.thr_std = config.params_dict['thr_std']

        self.model_outputs = max(config.params_dict['num_placed'])

        self.DA_mode = config.params_dict['DA_mode']
        if self.DA_mode:
            self.lutnum = config.params_dict['random_lutnum']
            self.convert_features = nn.Linear(config.params_dict['nodeperchip'], config.params_dict['feature_num'])

            self.test_lutnum = config.params_dict['lutnum']
            self.test_x_dist = torch.stack(
                [torch.linspace(-self.x_range, self.x_range, self.NodePerChip).to(self.device)
                 for _ in range(self.test_lutnum * self.NodePerChip)], dim=1)

        """
        Gaussian Distribution
        """
        self.dist_x = DiagGaussian(self.feature_num, self.model_outputs, self.x_range, self.thr_std)
        self.dist_y = DiagGaussian(self.feature_num, self.model_outputs, self.x_range, self.thr_std)

        baseline_x = torch.linspace(-self.x_range, self.x_range, self.x_dim)
        self.x_dist = torch.stack(
            [torch.FloatTensor(baseline_x).to(self.device)
             for _ in range(self.lutnum * self.model_outputs)], dim=1)
        baseline_y = torch.linspace(-self.x_range, self.x_range, self.y_dim)
        self.y_dist = torch.stack(
            [torch.FloatTensor(baseline_y).to(self.device)
             for _ in range(self.lutnum * self.model_outputs)], dim=1)

        self.plot_dist = torch.stack(
            [torch.linspace(-self.x_range - 0.1, self.x_range + 0.1, 200).to(self.device)
             for _ in range(self.lutnum * self.model_outputs)], dim=1)

        self.con_p = []
        self.discrete_x = []
        self.discrete_y = []

        self.grad_masking = [[], []]
        self.target = [[], []]

    def get_probs(self, action, idx=None):
        if idx is None:
            action_x, action_y = action[0], action[1]

            action_log_probs_x = torch.empty(action_x.shape).to(self.device)
            for i in range(action_x.shape[0]):
                action_log_probs_x[i] = self.discrete_x.gather(dim=1, index=action_x[i].unsqueeze(1)).squeeze()

            action_log_probs_y = torch.empty(action_y.shape).to(self.device)
            for i in range(action_y.shape[0]):
                action_log_probs_y[i] = self.discrete_y.gather(dim=1, index=action_y[i].unsqueeze(1)).squeeze()

            return [action_log_probs_x.detach(), action_log_probs_y.detach()]
        else:
            if idx == 0:
                action_log_probs_x = torch.empty(action.shape).to(self.device)
                for i in range(action.shape[0]):
                    action_log_probs_x[i] = self.discrete_x.gather(dim=1, index=action[i].unsqueeze(1)).squeeze()

                return action_log_probs_x.detach()
            else:
                action_log_probs_y = torch.empty(action.shape).to(self.device)
                for i in range(action.shape[0]):
                    action_log_probs_y[i] = self.discrete_y.gather(dim=1, index=action[i].unsqueeze(1)).squeeze()

                return action_log_probs_y.detach()

    def actor_layer(self, x, epochs, is_train):
        if self.DA_mode:
            x = F.relu(self.convert_features(x))

        dist_x = self.dist_x(x)
        dist_y = self.dist_y(x)

        discrete_dist_x = dist_x.log_probs(self.x_dist).exp().T
        discrete_dist_y = dist_y.log_probs(self.y_dist).exp().T

        discrete_dist_x = torch.softmax(discrete_dist_x, dim=1)
        discrete_dist_y = torch.softmax(discrete_dist_y, dim=1)

        cate_dist = Categorical(discrete_dist_x)
        action_x = torch.stack([cate_dist.sample() for _ in range(epochs)], dim=0)

        cate_dist = Categorical(discrete_dist_y)
        action_y = torch.stack([cate_dist.sample() for _ in range(epochs)], dim=0)

        ax = [torch.where(discrete_dist_x[i] == discrete_dist_x[i].max())[0][0].item()
              for i in range(len(discrete_dist_x))]
        ay = [torch.where(discrete_dist_y[i] == discrete_dist_y[i].max())[0][0].item()
              for i in range(len(discrete_dist_y))]

        action_x[-1] = torch.tensor(ax, dtype=torch.int64).to(self.device)
        action_y[-1] = torch.tensor(ay, dtype=torch.int64).to(self.device)

        action_log_probs_x = torch.empty(action_x.shape)
        for i in range(epochs):
            action_log_probs_x[i] = discrete_dist_x.gather(dim=1, index=action_x[i].unsqueeze(1)).squeeze()

        action_log_probs_y = torch.empty(action_y.shape)
        for i in range(epochs):
            action_log_probs_y[i] = discrete_dist_y.gather(dim=1, index=action_y[i].unsqueeze(1)).squeeze()

        self.discrete_x = discrete_dist_x.to(self.device)
        self.discrete_y = discrete_dist_y.to(self.device)

        return [action_x, action_y], [action_log_probs_x, action_log_probs_y]

    def actor_critic_layer(self, x, action, idx):
        if self.DA_mode:
            x = F.relu(self.convert_features(x))

        if idx is None:
            dist_x = self.dist_x(x)
            dist_y = self.dist_y(x)

            discrete_dist_x = dist_x.log_probs(self.x_dist).exp().T
            discrete_dist_y = dist_y.log_probs(self.y_dist).exp().T

            discrete_dist_x = torch.softmax(discrete_dist_x, dim=1)
            discrete_dist_y = torch.softmax(discrete_dist_y, dim=1)

            action_x, action_y = action[0], action[1]

            action_log_probs_x = torch.empty(action_x.shape)
            for i in range(action_x.shape[0]):
                action_log_probs_x[i] = discrete_dist_x.gather(dim=1, index=action_x[i].unsqueeze(1)).squeeze()

            action_log_probs_y = torch.empty(action_y.shape)
            for i in range(action_y.shape[0]):
                action_log_probs_y[i] = discrete_dist_y.gather(dim=1, index=action_y[i].unsqueeze(1)).squeeze()

            dist_entropy_x = dist_x.entropy().mean()
            dist_entropy_y = dist_y.entropy().mean()

            return [action_log_probs_x, action_log_probs_y], [dist_entropy_x, dist_entropy_y]
        else:
            if idx == 0:
                dist_x = self.dist_x(x)
                discrete_dist = torch.softmax(dist_x.log_probs(self.x_dist).exp().T, dim=1)
                dist_entropy = dist_x.entropy().mean()
            else:
                dist_y = self.dist_y(x)
                discrete_dist = torch.softmax(dist_y.log_probs(self.y_dist).exp().T, dim=1)
                dist_entropy = dist_y.entropy().mean()

            action_log_probs = torch.empty(action.shape).to(self.device)
            for i in range(action.shape[0]):
                action_log_probs[i] = discrete_dist.gather(dim=1, index=action[i].unsqueeze(1)).squeeze()

            return action_log_probs, dist_entropy

    def forward(self, x, epochs=0, actions=None, is_actor=False, is_train=True, idx=None):
        if is_actor:
            return self.actor_layer(x, epochs, is_train)
        else:
            return self.actor_critic_layer(x, actions, idx)


class GCN(nn.Module):
    def __init__(self, nfeat, dropout, MaxNum):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 32)
        self.gc2 = GraphConvolution(32, 16)
        self.gc3 = GraphConvolution(16, 1)

        self.readout = nn.Linear(MaxNum, 1)

        self.dropout = dropout

    def forward(self, x, adj, is_features=False):
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = F.dropout(F.relu(self.gc2(x, adj)), self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))

        if is_features:
            return x.squeeze()
        else:
            return self.readout(x.squeeze())
