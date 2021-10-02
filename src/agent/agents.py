from math import log
import torch
import numpy as np

from typing import Tuple, List

from agent.actor import Actor
from env.core_mapper import CoreMapper


class PPOAgent:
    def __init__(self, data) -> None:
        """Iniitialize PPO agent."""
        super(PPOAgent, self).__init__()

        self.rl_config = data.config["RL_config"]

        self.use_cuda = self.rl_config["use_cuda"]
        self.device = torch.device("cpu")
        if self.use_cuda:
            self.device = torch.device(f"cuda:{self.rl_config['device']}")
        
        # parameters
        self.batch_size = 256
        self.ppo_epoch = 32
        self.ppo_clip = 0.1
        self.lr = 0.001

        # environment and network model
        self.env = CoreMapper(data)
        self.policy = Actor(data)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # store episodes
        self.actions: List[List[int]] = []
        self.log_probs: List[List[int]] = []
        self.rewards: List[float] = []
        
    def reset_buffer(self) -> None:
        """Reset memory."""
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
    
    def save_epiosde(
        self,
        action: Tuple[List[int], List[int]],
        log_prob: Tuple[List[int], List[int]],
        reward: float
    ) -> None:
        """Store episode in buffer."""
        self.actions.append(
            (action[0].cpu().tolist(), action[1].cpu().tolist())
        )
        self.log_probs.append(
            (log_prob[0].cpu().tolist(), log_prob[1].cpu().tolist())
        )
        self.rewards.append(reward)

    def train(self) -> None:
        """Train agent."""
        best_reward = 1e12

        epoch = 1
        while True:
            self.reset_buffer()
            for _ in range(self.batch_size):
                # reward sampling
                with torch.no_grad():
                    action, log_prob = self.policy()
                reward = self.env.step(action)

                # store episodes
                self.save_epiosde(action, log_prob, reward)
                best_reward = min(-reward, -best_reward)
            
            print(f"# of epochs: {epoch}, reward_mean: {-np.mean(self.rewards):.2f} ({best_reward:.2f})")
            self.update_policy()

            epoch += 1

    def update_policy(self) -> None:
        """Update policy network."""
        actions = torch.tensor(self.actions).permute(1, 0, 2).to(self.device)
        log_probs = torch.FloatTensor(self.log_probs).permute(1, 0, 2).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)

        # normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-12)
        rewards = torch.clamp(rewards, -10, 10).unsqueeze(1).detach()

        # update policy network
        for _ in range(self.ppo_epoch):
            pi = self.policy(actions=actions)

            ratio_x = torch.exp(torch.log(pi[0]) - torch.log(log_probs[0]))
            ratio_y = torch.exp(torch.log(pi[1]) - torch.log(log_probs[1]))

            surr1_x = ratio_x * rewards
            surr2_x = torch.clamp(ratio_x, 1 - self.ppo_clip, 1 + self.ppo_clip) * rewards
            actor_loss_x = -torch.min(surr1_x, surr2_x).mean()

            surr1_y = ratio_y * rewards
            surr2_y = torch.clamp(ratio_y, 1 - self.ppo_clip, 1 + self.ppo_clip) * rewards
            actor_loss_y = -torch.min(surr1_y, surr2_y).mean()

            loss = actor_loss_x + actor_loss_y

            self.policy.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
