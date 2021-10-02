import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from numbers import Number


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x, is_bias=False):
        bias = self._bias.t().view(1, -1)
        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# Normal
class FixedNormal(torch.distributions.Normal):
    def get_logprob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def log_probs(self, actions):
        return self.get_logprob(actions)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, x_range):
        super(DiagGaussian, self).__init__()

        self.x_range = x_range
        self.multiplex = num_outputs

        hidden_layer = 512
        self.read_out = nn.Sequential(
            nn.Linear(num_inputs, hidden_layer),
        )

        self.fc_mean = nn.Sequential(
            nn.Linear(hidden_layer, num_outputs),
        )

        self.fc_std = nn.Sequential(
            nn.Linear(hidden_layer, num_outputs),
        )
        self.init_w()

    def init_ws(self, model):
        for name, param in model.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            if param.dim() > 1 and 'weight' in name:
                nn.init.orthogonal_(param)

    def init_w(self):
        self.read_out.apply(self.init_ws)
        self.fc_mean.apply(self.init_ws)
        self.fc_std.apply(self.init_ws)

    def forward(self, x):
        x = torch.tanh(self.read_out(x))

        action_mean = torch.tanh(self.fc_mean(x)) * self.x_range
        action_std = F.softplus(self.fc_std(x))

        return FixedNormal(action_mean, torch.sqrt(action_std))


# Beta
class FixedBeta(torch.distributions.Beta):
    def log_probs(self, actions):
        return super().log_prob(actions)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagBeta(nn.Module):
    def __init__(self, num_inputs, num_outputs, x_range):
        super(DiagBeta, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.x_range = x_range
        self.multiplex = num_outputs
        self.fc_alpha = init_(nn.Linear(num_inputs, num_outputs))
        self.fc_beta = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        alpha = F.softplus(self.fc_alpha(x)) + 1
        beta = F.softplus(self.fc_beta(x)) + 1

        return FixedBeta(alpha, torch.sqrt(beta))
