import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import shutil
import pandas as pd
import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from Code.Environment import make_env
from Code.Model import Actor

import Code.Utils as utils
import Code.Memory as memory


class Agent:
    def __init__(self, config):
        super(Agent, self).__init__()

        self.use_cuda = config.use_cuda
        self.device = config.device

        self.mode = config.params_dict['mode']

        self.lutnum = config.params_dict['lutnum']
        self.x_dim = config.params_dict['x_dim']
        self.y_dim = config.params_dict['y_dim']
        self.x_range = config.params_dict['x_range']
        self.thr_std = config.params_dict['thr_std']

        self.DA_mode = config.params_dict['DA_mode']

        self.stop_ratio = config.params_dict['stop_ratio']
        self.init_epochs = config.params_dict['init_epochs']
        self.max_epoch = config.params_dict['max_epochs']

        self.memory_buffer_size = config.params_dict['memory_buffer_size']
        self.batch_size = config.params_dict['batch_size']
        self.mini_batch_size = config.params_dict['mini_batch_size']
        self.ppo_epoch = config.params_dict['ppo_epoch']
        self.ppo_clip = config.params_dict['ppo_clip']
        self.entropy_coef = config.params_dict['entropy_coef']

        self.allnodenum = config.params_dict['allnodenum']

        self.r_range = config.params_dict['r_range']
        self.feature_num = config.params_dict['feature_num']

        self.load_pre_trained = config.params_dict['load_pre_trained']

        self.EntireLUT = config.params_dict['lut']
        self.train_filename = config.params_dict['lut_filename_e']

        self.train_lut = config.params_dict['lut']
        self.folder_name = config.params_dict['ProjectFolderPath']
        self.plot_folder = config.params_dict['plot_folder_path']
        self.sub_folder = config.params_dict['sub_folder_path']

        self.print_interval = config.params_dict['print_interval']
        self.save_interval = config.params_dict['save_interval']

        self.train_env = make_env(config)

        self.policy = Actor(config)
        self.copy_policy = Actor(config)

        if self.use_cuda:
            self.policy = self.policy.to(self.device)

        self.lr = config.params_dict['policy_lr']

        self.policy_optimizer_x = torch.optim.Adam(self.policy.dist_x.parameters(), lr=self.lr)
        self.policy_optimizer_y = torch.optim.Adam(self.policy.dist_y.parameters(), lr=self.lr)

        self.policy_optimizer = torch.optim.Adam(
            [
                {"params": self.policy.dist_x.fc_mean.parameters(), "lr": self.lr},
                {"params": self.policy.dist_x.fc_std.parameters(), "lr": self.lr},
                {"params": self.policy.dist_y.fc_mean.parameters(), "lr": self.lr},
                {"params": self.policy.dist_y.fc_std.parameters(), "lr": self.lr},
            ]
        )

        self.policy_optimizer_mean = torch.optim.Adam(
            [
                {"params": self.policy.dist_x.fc_mean.parameters(), "lr": self.lr},
                {"params": self.policy.dist_y.fc_mean.parameters(), "lr": self.lr},
            ]
        )
        self.lr_scaling = 1
        self.policy_optimizer_std = torch.optim.Adam(
            [
                {"params": self.policy.dist_x.fc_std.parameters(), "lr": self.lr * self.lr_scaling},
                {"params": self.policy.dist_y.fc_std.parameters(), "lr": self.lr * self.lr_scaling},
            ]
        )

        self.policy_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.policy_optimizer,
                                                                     milestones=[300], gamma=0.1)
        self.policy_scheduler_x = torch.optim.lr_scheduler.MultiStepLR(self.policy_optimizer_x,
                                                                       milestones=[2000], gamma=0.5)
        self.policy_scheduler_y = torch.optim.lr_scheduler.MultiStepLR(self.policy_optimizer_y,
                                                                       milestones=[4000], gamma=0.5)

        # self.policy_optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=self.lr)
        # self.policy_optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.lr, momentum=0.99)
        # self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.lr)

        # Extract Features
        if config.params_dict['DA_mode']:
            self.lutnum = config.params_dict['random_lutnum']
            self.input_features = config.params_dict['random_features'].to(self.device)
            self.da_policy = Actor(config).to(self.device)

            self.test_lutnum = config.params_dict['lutnum']
            self.test_input_features = torch.FloatTensor(config.params_dict['test_features']).to(self.device)
        else:
            # self.input_features = torch.zeros([self.lutnum, self.feature_num]).to(self.device)
            # self.input_features = torch.ones([self.lutnum, self.feature_num]).to(self.device) * 0.01
            self.input_features = torch.rand([self.lutnum, self.feature_num]).to(self.device) * 0.01

            np.savetxt(self.sub_folder + "/input_features.csv", np.array(self.input_features.cpu()))

        # Load pre-trained
        if self.load_pre_trained:
            model = torch.load(self.folder_name + '/model.pt')
            self.policy.load_state_dict(model['policy_model'], strict=False)

            if not self.DA_mode:
                self.input_features = torch.FloatTensor(np.loadtxt(self.folder_name + '/input_features.csv')[0])

        self.memory = memory.Memory(config)

        self.switch_cnt = 0
        self.loss_switch = 1
        # torch.autograd.set_detect_anomaly(True)

    def init_network(self):
        # print("\n  ==>  WITH NETWORK PLACEMENTS ...\n")
        best_rewards = [1e12 for _ in range(self.lutnum)]
        rewards = [[] for _ in range(self.lutnum)]

        action, _ = self.policy(self.input_features, self.init_epochs, is_actor=True)

        if self.DA_mode:
            reward, _ = self.train_env.step(action, is_random=True)
        else:
            reward, _ = self.train_env.step(self.init_epochs, action)

        for i in range(self.lutnum):
            rewards[i] += reward[i]

            if -max(reward[i]) < best_rewards[i]:
                best_rewards[i] = -max(reward[i])

        reward = np.sum(reward, axis=0) / self.lutnum
        self.save_eps(action[0].tolist(), action[1].tolist(), reward.tolist(), [], [])

        mean_rwd = np.array([np.mean(i) for i in rewards])
        std_rwd = np.array([np.std(i) for i in rewards])

        # print("Best (Objective_Func) of random placement: {:.2f}".format(np.mean(best_rewards)),
        #       "mean: {:.2f},".format(np.mean(-mean_rwd)),
        #       "std: {:.2f})\n".format(np.mean(std_rwd)))

        self.memory.initUpdateRwd(self.init_epochs, mean_rwd, std_rwd, best_rewards)

    def save_eps(self, action_x, action_y, reward, prob_x, prob_y):
        self.memory.save_eps(action_x, action_y, reward, prob_x, prob_y)

    def train_network(self):
        best_rewards = [1e12 for _ in range(self.lutnum)]

        mean_rwd, std_rwd = self.memory.getMeanStd()

        for epoch in range(1, self.max_epoch):
            # if epoch > 500:
            #     self.batch_size = 512
            #     self.ppo_epoch = 16

            action, action_log_prob = self.policy(self.input_features, self.batch_size, is_actor=True)

            if self.DA_mode:
                reward, realpos = self.train_env.step(action, is_random=True)
            else:
                reward, realpos = self.train_env.step(self.batch_size, action)

            action = torch.tensor(realpos, dtype=torch.int64)

            for i in range(self.lutnum):
                if -max(reward[i]) < best_rewards[i]:
                    best_rewards[i] = -max(reward[i])

                    # placement = self.train_env.BestPlacement[i].squeeze()
                    # print("\n* Best Rewards: {:.2f}".format(best_rewards[i]),
                    #       "-- num_placed: {}\n".format((placement != 0).sum()),
                    #       "\n{}\n".format(placement))
            rewards = reward

            reward = np.sum(reward, axis=0) / self.lutnum

            self.save_eps(action[0].tolist(), action[1].tolist(), reward.tolist(), action_log_prob[0].cpu().tolist(), action_log_prob[1].cpu().tolist())

            self.memory.updateRwdParameters(self.batch_size, rewards)

            self.update_network()
            self.memory.clearEps()

            self.save_datas(epoch, rewards, best_rewards)

            if self.memory.sample_taken > 151000:
                self.save_datas(epoch, rewards, best_rewards, sav=True)

                rwds = self.memory.rewards_curve.copy()

                x_list = list(range(1, len(rwds) + 1))

                plt.ylim([0, max(rwds)])
                plt.plot(x_list, rwds)
                plt.savefig(self.memory.rewards_curve_fig)
                plt.close()

                while len(rwds) % 1000 != 0:
                    rwds.append(0)
                datas = pd.DataFrame(np.array(rwds).reshape(int(len(rwds) / 1000), 1000))
                datas.to_csv(self.memory.rewards_curve_path, header=None, index=None)

                break

        print(" # NUMBERS:{}".format(self.sub_folder))
        print(
            "(Best_reward): {},\n".format(min(best_rewards)),
            "(feature_num): {}\n".format(self.feature_num),
            "(ppo_clip): {}\n".format(self.ppo_clip),
            "(policy_lr): {}\n".format(self.lr),
            "(memory_buffer_size): {},\n".format(self.memory_buffer_size),
            "(batch_size): {},\n".format(self.batch_size),
            "(mini_batch_size): {},\n".format(self.mini_batch_size),
            "(ppo_epoch): {},\n".format(self.ppo_epoch),
            "(x_range - thr_std): {} - {}\n".format(self.x_range, self.thr_std),
            "(r_range): {}\n".format(self.r_range),
            "(samples): {}\n".format(self.memory.sample_taken),
        )

        if min(best_rewards) < 14000:
            print("\nFIND...?")
            exit()

    def update_network(self):
        # mean_rwd, std_rwd = self.memory.getMeanStd()
        mean_rwd, std_rwd = np.mean(self.memory.rewards), np.std(self.memory.rewards)
        min_rwd, max_rwd = np.min(self.memory.rewards), np.max(self.memory.rewards)

        self.copy_policy.discrete_x = self.policy.discrete_x
        self.copy_policy.discrete_y = self.policy.discrete_y
        for ppo_epoch in range(self.ppo_epoch):
            actions, rewards = self.memory.getUpdateEps(self.mini_batch_size)

            actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

            rewards = (rewards - mean_rwd) * self.r_range / std_rwd
            rewards = torch.clamp(rewards, -self.r_range, self.r_range)

            rewards = rewards.detach()

            """
            Update individually (x, y)
            """
            # if self.loss_switch == 1:
            #     probs = self.copy_policy.get_probs(actions[0], idx=0)
            #     pi, dist_entropy = self.policy(self.input_features, actions=actions[0], idx=0)
            #
            #     ratio_x = torch.exp(torch.log(pi) - torch.log(probs))
            #
            #     surr1_x = ratio_x * rewards
            #     surr2_x = torch.clamp(ratio_x, 1 - self.ppo_clip, 1 + self.ppo_clip) * rewards
            #     actor_loss_x = -torch.min(surr1_x, surr2_x).mean()
            #
            #     loss = actor_loss_x + self.entropy_coef * dist_entropy
            #
            #     self.policy.dist_x.zero_grad()
            #     loss.backward()
            #     self.policy_optimizer_x.step()
            # else:
            #     probs = self.copy_policy.get_probs(actions[1], idx=1)
            #     pi, dist_entropy = self.policy(self.input_features, actions=actions[1], idx=1)
            #
            #     ratio_y = torch.exp(torch.log(pi) - torch.log(probs))
            #
            #     surr1_y = ratio_y * rewards
            #     surr2_y = torch.clamp(ratio_y, 1 - self.ppo_clip, 1 + self.ppo_clip) * rewards
            #     actor_loss_y = -torch.min(surr1_y, surr2_y).mean()
            #
            #     loss = actor_loss_y + self.entropy_coef * dist_entropy
            #
            #     self.policy.dist_y.zero_grad()
            #     loss.backward()
            #     self.policy_optimizer_y.step()

            """
            Update individually (mean, std)
            """
            # probs = self.copy_policy.get_probs(actions)
            #
            # pi, dist_entropy = self.policy(self.input_features, actions=actions)
            #
            # ratio_x = torch.exp(torch.log(pi[0]) - torch.log(probs[0]))
            # ratio_y = torch.exp(torch.log(pi[1]) - torch.log(probs[1]))
            #
            # surr1_x = ratio_x * rewards
            # surr2_x = torch.clamp(ratio_x, 1 - self.ppo_clip, 1 + self.ppo_clip) * rewards
            # actor_loss_x = -torch.min(surr1_x, surr2_x).mean()
            #
            # surr1_y = ratio_y * rewards
            # surr2_y = torch.clamp(ratio_y, 1 - self.ppo_clip, 1 + self.ppo_clip) * rewards
            # actor_loss_y = -torch.min(surr1_y, surr2_y).mean()
            #
            # loss = actor_loss_x + actor_loss_y + self.entropy_coef * sum(dist_entropy)
            #
            # self.policy.zero_grad()
            # loss.backward()
            #
            # self.policy_optimizer_mean.step()
            # self.policy_optimizer_std.step()

            """
            Update simultaneously
            """
            probs = self.copy_policy.get_probs(actions)

            pi, dist_entropy = self.policy(self.input_features, actions=actions)

            ratio_x = torch.exp(torch.log(pi[0]) - torch.log(probs[0]))
            ratio_y = torch.exp(torch.log(pi[1]) - torch.log(probs[1]))

            surr1_x = ratio_x * rewards
            surr2_x = torch.clamp(ratio_x, 1 - self.ppo_clip, 1 + self.ppo_clip) * rewards
            actor_loss_x = -torch.min(surr1_x, surr2_x).mean()

            surr1_y = ratio_y * rewards
            surr2_y = torch.clamp(ratio_y, 1 - self.ppo_clip, 1 + self.ppo_clip) * rewards
            actor_loss_y = -torch.min(surr1_y, surr2_y).mean()

            loss = actor_loss_x + actor_loss_y + self.entropy_coef * sum(dist_entropy)

            self.policy.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

        self.switch_cnt += 1

        if self.switch_cnt == 1:
            self.loss_switch *= -1
            self.switch_cnt = 0

    def save_datas(self, epoch, rewards, best_rewards, sav=False):
        rewards = -np.array(rewards)
        if epoch % self.print_interval == 0:
            strepoch = str(epoch).zfill(6)
            # epoch * self.batch_size
            print(" # of {} epoch ({} samples)".format(strepoch, self.memory.sample_taken),
                  "ave_reward: {:.2f} (std: {:.2f})".format(np.mean(rewards), np.std(rewards)), end=' ----- ( ')
            if self.DA_mode:
                self.Domain_Adaptation()
            else:
                for lut_idx in range(self.lutnum):
                    print(
                        "{}: {} k+ ({:.2f} k+)".format(self.train_filename[lut_idx], int(np.mean(rewards[lut_idx]) / 1000),
                                                   best_rewards[lut_idx] / 1000), end=' ')
                print(')')

        """
            Plot Distribution
        """
        if epoch % 5000000 == 0 or sav:
            x = self.input_features

            dist_x = self.policy.dist_x(x)
            dist_y = self.policy.dist_y(x)

            x_prob_r = dist_x.log_probs(self.policy.x_dist).exp().detach().cpu().numpy()
            y_prob_r = dist_y.log_probs(self.policy.y_dist).exp().detach().cpu().numpy()
            x_prob = dist_x.log_probs(self.policy.plot_dist).exp().detach().cpu().numpy()
            y_prob = dist_y.log_probs(self.policy.plot_dist).exp().detach().cpu().numpy()

            plt.rcParams["figure.figsize"] = (9, 12)
            ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2, colspan=2)

            grid = np.zeros([self.y_dim, self.x_dim], dtype=np.int64)

            x_grid, y_grid = x_prob_r.T, y_prob_r.T
            ax = [np.where(x_grid[i] == x_grid[i].max())[0][0] for i in range(len(x_grid))]
            ay = [np.where(y_grid[i] == y_grid[i].max())[0][0] for i in range(len(x_grid))]
            rr, realpos = self.train_env.step(1, [ax, ay])
            for i in range(len(realpos[0][0])):
                grid[realpos[1][0][i]][realpos[0][0][i]] = i + 1

            color = sns.color_palette("husl", self.policy.model_outputs + 1)
            color[0] = (0.8, 0.8, 0.8)

            heatmap_labels = np.zeros(grid.shape, dtype=np.int)
            for i in range(self.y_dim):
                for j in range(self.x_dim):
                    if grid[i][j] != 0:
                        heatmap_labels[i][j] = self.train_env.vec_placed[0][grid[i][j] - 1]
            sns.heatmap(grid, annot=heatmap_labels, cmap=color, cbar=False, linewidths=.5, fmt='g')

            plt.title('reward: {:.2f}'.format(-rr[0][0]))

            y_limit = max(np.max(x_prob), np.max(y_prob))
            x_prob_r = np.where(x_prob_r > y_limit, y_limit, x_prob_r)
            y_prob_r = np.where(y_prob_r > y_limit, y_limit, y_prob_r)
            x_prob = np.where(x_prob > y_limit, y_limit, x_prob)
            y_prob = np.where(y_prob > y_limit, y_limit, y_prob)

            plt.subplot(3, 2, 5)
            plt.xlim([-self.x_range * 1.1, self.x_range * 1.1])
            plt.ylim([0, y_limit + 0.1])
            for i in range(x_prob.shape[1]):
                plt.plot(self.policy.plot_dist.detach().numpy()[:, i], x_prob[:, i], color=color[i + 1])
                plt.plot(self.policy.x_dist.detach().numpy()[:, i], x_prob_r[:, i], '*', color=color[i + 1])

            plt.subplot(3, 2, 6)
            plt.xlim([-self.x_range * 1.1, self.x_range * 1.1])
            plt.ylim([0, y_limit + 0.1])
            for i in range(y_prob.shape[1]):
                plt.plot(self.policy.plot_dist.detach().numpy()[:, i], y_prob[:, i], color=color[i + 1])
                plt.plot(self.policy.y_dist.detach().numpy()[:, i], y_prob_r[:, i], '*', color=color[i + 1])

            plt.savefig(self.plot_folder + '/{}.png'.format(epoch))
            plt.close()




            # torch.save({
            #     'policy_model': self.policy.state_dict(),
            #     'optimizer': self.policy_optimizer.state_dict()
            # }, self.sub_folder + '/model_{}.pt'.format(epoch))

            # if len(self.r_trace) > 100:
            #     while len(self.r_trace) % 100 != 0:
            #         self.r_trace.append(0)
            # datas = pd.DataFrame(np.array(self.r_trace).reshape(int(len(self.r_trace) / 100), 100))
            # datas.to_csv(self.folder + "\\rewards_record.csv", header=None, index=None)

            # plt.plot(np.linspace(0, len(self.r_trace) - 1, len(self.r_trace)), self.r_trace)
            # plt.savefig("rewards.png")

    def Domain_Adaptation(self):
        self.da_policy.load_state_dict(self.policy.state_dict())
        self.da_policy.eval()

        action = self.da_policy(self.test_input_features, is_actor=True, is_train=False)
        rewards, _ = self.train_env.step(action)

        for lut_idx in range(self.test_lutnum):
            print("{}: {} k+ ".format(self.train_filename[lut_idx], -int(rewards[lut_idx] / 1000)), end=' ')
        print(')')
