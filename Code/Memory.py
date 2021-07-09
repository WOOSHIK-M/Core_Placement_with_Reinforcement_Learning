import torch
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

import Code.Utils as utils


class Memory:
    def __init__(self, config):
        super(Memory, self).__init__()

        self.use_cuda = config.use_cuda
        self.device = config.device

        self.sub_folder_path = config.params_dict['sub_folder_path']

        self.Epi_buffer_size = config.params_dict['memory_buffer_size']

        self.save_interval = config.params_dict['save_interval']
        self.save_cnt = 0

        # Save Episode
        self.actions_x = []
        self.actions_y = []
        self.rewards = []
        self.probs_x = []
        self.probs_y = []

        # Save reward
        self.total_reward_num = 0
        self.mean_rwd = []
        self.std_rwd = []
        self.best_reward = []

        self.sample_taken = 0

        self.rewards_curve = []

        self.rewards_curve_fig = self.sub_folder_path + '/rewards_curve.png'
        self.rewards_curve_path = self.sub_folder_path + '/rewards_curve.csv'

    def save_eps(self, action_x, action_y, reward, prob_x, prob_y):
        for i in range(len(action_x)):
            temp_action_x = action_x[i]
            temp_action_y = action_y[i]
            if temp_action_x not in self.actions_x \
                    or temp_action_y not in self.actions_y:
                self.actions_x.append(temp_action_x)
                self.actions_y.append(temp_action_y)
                self.rewards.append(reward[i])

                self.sample_taken += 1
                self.rewards_curve.append(-reward[i])

        """
        Plot Rewards curve
        """
        # self.save_cnt += 1
        #
        # if self.save_cnt % self.save_interval == 0:
        #     rwds = self.rewards_curve.copy()
        #
        #     x_list = list(range(1, len(rwds) + 1))
        #
        #     plt.ylim([0, max(rwds)])
        #     plt.plot(x_list, rwds)
        #     plt.savefig(self.rewards_curve_fig)
        #     plt.close()
        #
        #     while len(rwds) % 1000 != 0:
        #         rwds.append(0)
        #     datas = pd.DataFrame(np.array(rwds).reshape(int(len(rwds) / 1000), 1000))
        #     datas.to_csv(self.rewards_curve_path, header=None, index=None)
        #
        #     self.save_cnt = 0

    def clearEps(self):
        clearValue = max(len(self.actions_x) - self.Epi_buffer_size, 0)

        newseq = np.argsort(self.rewards)
        self.actions_x = np.array(self.actions_x)[newseq].tolist()
        self.actions_y = np.array(self.actions_y)[newseq].tolist()
        self.rewards = np.array(self.rewards)[newseq].tolist()

        del self.actions_x[:clearValue]
        del self.actions_y[:clearValue]
        del self.rewards[:clearValue]

    def getUpdateEps(self, batch_size):
        train_size = len(self.actions_x)
        batch_mask = np.random.choice(train_size, batch_size)

        return [np.array(self.actions_x)[batch_mask], np.array(self.actions_y)[batch_mask]], \
               np.array(self.rewards)[batch_mask]

    def initUpdateRwd(self, init_epoch, mean_rwd, std_rwd, best_reward):
        self.total_reward_num += init_epoch
        self.mean_rwd = mean_rwd
        self.std_rwd = std_rwd
        self.best_reward = best_reward

    def getMeanStd(self):
        return self.mean_rwd, self.std_rwd

    def updateRwdParameters(self, new_num, rewards):
        for idx, reward in enumerate(rewards):
            self.mean_rwd[idx] = (self.mean_rwd[idx] * self.total_reward_num + np.sum(reward)) \
                            / (self.total_reward_num + new_num)
            self.std_rwd[idx] = math.sqrt(((math.pow(self.std_rwd[idx], 2) * self.total_reward_num) + (np.std(reward) * new_num))
                                     / self.total_reward_num + new_num)
        self.total_reward_num += new_num
