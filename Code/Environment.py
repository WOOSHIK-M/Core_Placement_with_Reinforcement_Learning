import numpy as np
import random
import copy
import time
import networkx as nx

import Code.Utils as utils
import Code.Reward_funcs as reward_funcs


class CoreMapper:
    def __init__(self, config):
        super(CoreMapper, self).__init__()

        self.use_cuda = config.use_cuda
        self.device = config.device

        self.lutnum = config.params_dict['lutnum']

        self.x_dim = config.params_dict['x_dim']
        self.y_dim = config.params_dict['y_dim']
        self.X_DIM = config.params_dict['X_DIM']
        self.Y_DIM = config.params_dict['Y_DIM']

        self.allnodenum = config.params_dict['allnodenum']
        self.allchipnum = config.params_dict['allchipnum']
        self.NodePerChip = config.params_dict['nodeperchip']

        self.DA_mode = config.params_dict['DA_mode']
        self.random_lutnum = config.params_dict['random_lutnum']
        self.random_nodes = config.params_dict['random_nodes']

        self.num_placed = max(config.params_dict['num_placed'])
        self.vec_placed = config.params_dict['vec_placed']

        self.x_range = config.params_dict['x_range']

        self.G = config.params_dict['G'][0]

        # zigzag
        self.phy_x = config.params_dict['phy_x']
        self.phy_y = config.params_dict['phy_y']

        # neighbor
        self.phy_x_neighbor = np.where(np.array(self.phy_y) % 2 == 0, np.array(self.phy_x),
                                       self.x_dim - np.array(self.phy_x) - 1)
        self.phy_y_neighbor = self.phy_y

        # Random!
        self.phy_x_random, self.phy_y_random = self.make_random()

        # Circle
        self.phy_x_circle, self.phy_y_circle = self.make_circle()

        # Rectangle
        self.phy_x_rectangle, self.phy_y_rectangle = self.make_rectangle()

        # Assign placement coordinate
        self.placed_pos_x = self.phy_x_neighbor
        self.placed_pos_y = self.phy_y_neighbor

        self.search_table = np.zeros([self.y_dim * self.Y_DIM, self.x_dim * self.X_DIM], dtype=np.int)
        for i in range(self.allnodenum):
            self.search_table[self.placed_pos_y[i], self.placed_pos_x[i]] = i

        self.node = config.params_dict['nodes']

        self.deadlock_control = config.params_dict['deadlock_control']
        self.deadlock_coef = config.params_dict['deadlock_coef']
        self.deadlockCons = []
        for node_i in self.node:
            temp_cons = 0
            for temp_node in node_i:
                if temp_node.is_mul == 1:
                    temp_cons += 1
            self.deadlockCons.append(temp_cons)

        if self.allchipnum == 1:
            # self.chipidx = np.ones(self.num_placed, dtype=np.int)
            placedlength = [len(i) for i in self.vec_placed]
            self.chipidx = np.ones([self.lutnum, max(placedlength)], dtype=np.int)
        else:
            self.chipidx = np.array(config.params_dict['assignedchipidx'], dtype=np.int)

        self.nodeidx = []

        # self.ChipPlacement = [[] for _ in range(self.lutnum)]
        self.BestPlacement = []

        # node_degree = [self.G.in_degree[i + 1] + self.G.out_degree[i + 1] for i in self.vec_placed[0]]
        # new_seq = np.argsort(node_degree)[::-1]

        # new_vec = []
        # for i in new_seq:
        #     new_vec.append(self.vec_placed[0][i])
        # self.vec_placed[0] = new_vec

        # nodepos = new_seq[0]
        # tempnode = self.vec_placed[0][0]
        # self.vec_placed[0][0] = self.vec_placed[0][nodepos]
        # self.vec_placed[0][nodepos] = tempnode
        self.sx, self.sy = self.phy_x_rectangle[0], self.phy_y_rectangle[0]

        self.overlap_pri = list(range(self.NodePerChip))

    def make_random(self):
        RS_list = np.linspace(0, self.allnodenum - 1, self.allnodenum, dtype=np.int)
        random.shuffle(RS_list)
        phy_x, phy_y = [], []
        for i in range(self.allnodenum):
            phy_x.append(self.phy_x[RS_list[i]])
            phy_y.append(self.phy_y[RS_list[i]])
        return phy_x, phy_y

    def make_circle(self):
        phy_x, phy_y = [], []

        chip_extracted = np.zeros([self.y_dim, self.x_dim])

        sx, sy = int(self.x_dim / 2), int(self.y_dim / 2)
        if self.x_dim % 2 == 0:
            sx -= 1

        for i in range(self.allnodenum):
            tempx, tempy = sx, sy
            distance = 1
            if chip_extracted[tempy, tempx] != 0:
                while True:
                    # Go up
                    tempy -= 1
                    tx, ty = self.detect_circle(distance, tempx, tempy, chip_extracted)
                    # tx, ty = self.detect_rectangular(distance, tempx, tempy, chip_extracted)

                    if tx != -1:
                        tempx, tempy = tx, ty
                        break
                    else:
                        distance += 1
            phy_x.append(tempx)
            phy_y.append(tempy)
            chip_extracted[tempy, tempx] = i + 1
        return phy_x, phy_y

    def make_rectangle(self):
        phy_x, phy_y = [], []

        chip_extracted = np.zeros([self.y_dim, self.x_dim])

        sx, sy = int(self.x_dim / 2), int(self.y_dim / 2)
        if self.x_dim % 2 == 0:
            sx -= 1

        for i in range(self.allnodenum):
            tempx, tempy = sx, sy
            distance = 1
            if chip_extracted[tempy, tempx] != 0:
                while True:
                    # Go up
                    tempy -= 1
                    tx, ty = self.detect_rectangular(distance, tempx, tempy, chip_extracted)

                    if tx != -1:
                        tempx, tempy = tx, ty
                        break
                    else:
                        distance += 1
            phy_x.append(tempx)
            phy_y.append(tempy)
            chip_extracted[tempy, tempx] = i + 1
        return phy_x, phy_y

    def detect_random(self, chip_extracted):
        rs_list = list(range(0, self.allnodenum))
        random.shuffle(rs_list)

        tempx, tempy = self.placed_pos_x[rs_list[0]], self.placed_pos_y[rs_list[0]]

        while chip_extracted[tempy, tempx] != 0:
            random.shuffle(rs_list)
            tempx, tempy = self.placed_pos_x[rs_list[0]], self.placed_pos_y[rs_list[0]]
        return tempx, tempy

    def detect_circle(self, distance, tempx, tempy, chip_extracted):
        # Go right & down
        for _ in range(distance):
            tempx += 1
            tempy += 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy

        # Go left & down
        for _ in range(distance):
            tempx -= 1
            tempy += 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy

        # Go left & up
        for _ in range(distance):
            tempx -= 1
            tempy -= 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy

        # Go right & up
        for _ in range(distance):
            tempx += 1
            tempy -= 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy

        return -1, -1

    def detect_rectangular(self, distance, tempx, tempy, chip_extracted):
        # Go right & down
        for _ in range(distance):
            tempx += 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy
        for _ in range(distance):
            tempy += 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy

        # Go down & left
        for _ in range(distance):
            tempy += 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy
        for _ in range(distance):
            tempx -= 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy

        # Go left & up
        for _ in range(distance):
            tempx -= 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy
        for _ in range(distance):
            tempy -= 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy

        # Go up & right
        for _ in range(distance):
            tempy -= 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy
        for _ in range(distance):
            tempx += 1
            if 0 <= tempx < self.x_dim and 0 <= tempy < self.y_dim:
                if chip_extracted[tempy][tempx] == 0:
                    return tempx, tempy

        return -1, -1

    def step(self, batch_size, pos_value, is_random=False):
        if self.use_cuda:
            pos_value = [pos.cpu() for pos in pos_value]

        PlacementRecord = []

        rewards = [[] for _ in range(self.lutnum)]

        pos_x_all = np.array(pos_value[0], dtype=np.int).reshape([batch_size, self.lutnum, self.num_placed])
        pos_y_all = np.array(pos_value[1], dtype=np.int).reshape([batch_size, self.lutnum, self.num_placed])

        # pos_x_all[:, 0, 0] = self.sx
        # pos_y_all[:, 0, 0] = self.sy

        # PLACEMENT!
        realpos_x = []
        realpos_y = []
        for idx1 in range(batch_size):
            pos_x = pos_x_all[idx1]
            pos_y = pos_y_all[idx1]

            for compute_idx in range(pos_x.shape[0]):
                random.shuffle(self.overlap_pri)

                value_x = pos_x[compute_idx]
                value_y = pos_y[compute_idx]

                vec_place = self.vec_placed[compute_idx]
                chip_idx = self.chipidx[compute_idx]

                ChipPlacement = [np.zeros([self.y_dim, self.x_dim], dtype=np.int) for _ in range(self.allchipnum)]

                x_vec = [-1 for _ in range(self.allnodenum)]
                y_vec = [-1 for _ in range(self.allnodenum)]

                newpos_x = [0 for _ in range(self.num_placed)]
                newpos_y = [0 for _ in range(self.num_placed)]

                overlapped = [False for _ in range(self.num_placed)]
                for node_idx, chipidx in enumerate(chip_idx):
                    if node_idx < len(vec_place):
                        chipnumber = chipidx - 1

                        nodeidx = vec_place[node_idx]

                        innerchipx = value_x[node_idx]
                        innerchipy = value_y[node_idx]

                        if ChipPlacement[chipnumber][innerchipy, innerchipx] == 0:
                            x_vec[nodeidx] = innerchipx
                            y_vec[nodeidx] = innerchipy

                            ChipPlacement[chipnumber][innerchipy, innerchipx] = nodeidx + 1

                            newpos_x[node_idx] = innerchipx
                            newpos_y[node_idx] = innerchipy
                        else:
                            already_placed = ChipPlacement[chipnumber][innerchipy, innerchipx] - 1
                            if self.overlap_pri[nodeidx] > self.overlap_pri[already_placed]:
                                x_vec[nodeidx] = innerchipx
                                y_vec[nodeidx] = innerchipy

                                ChipPlacement[chipnumber][innerchipy, innerchipx] = nodeidx + 1

                                newpos_x[node_idx] = innerchipx
                                newpos_y[node_idx] = innerchipy

                                overlapped[np.where(already_placed == vec_place)[0][0]] = True
                            else:
                                overlapped[node_idx] = True

                # mapping overlapped cores
                for node_idx, chipidx in enumerate(chip_idx):
                    if node_idx < len(vec_place):
                        if overlapped[node_idx]:
                            chipnumber = chipidx - 1

                            nodeidx = vec_place[node_idx]

                            innerchipx = value_x[node_idx]
                            innerchipy = value_y[node_idx]

                            chip_extracted = ChipPlacement[chipnumber]

                            distance = 1
                            tempx = innerchipx
                            tempy = innerchipy

                            while True:
                                # Go up
                                tempy -= 1

                                # tx, ty = self.detect_random(chip_extracted)
                                tx, ty = self.detect_circle(distance, tempx, tempy, chip_extracted)
                                # tx, ty = self.detect_rectangular(distance, tempx, tempy, chip_extracted)

                                if tx != -1:
                                    tempx, tempy = tx, ty
                                    break
                                else:
                                    distance += 1

                            x_vec[nodeidx] = tempx
                            y_vec[nodeidx] = tempy

                            ChipPlacement[chipnumber][tempy, tempx] = nodeidx + 1

                            newpos_x[node_idx] = tempx
                            newpos_y[node_idx] = tempy

                realpos_x.append(newpos_x)
                realpos_y.append(newpos_y)

                rewards[compute_idx].append(self.reward(compute_idx, x_vec, y_vec, is_random))

                PlacementRecord.append(np.array(ChipPlacement, dtype=np.int))

        self.BestPlacement = [PlacementRecord[np.argsort(temp_r)[-1]] for temp_r in rewards]

        return rewards, [realpos_x, realpos_y]

    def reward(self, compute_idx, order_x, order_y, is_random):
        # return -reward_funcs.Communication_Cost(self.node[compute_idx],
        #                                         order_x,
        #                                         order_y,
        #                                         self.deadlock_control,
        #                                         self.deadlock_coef,
        #                                         self.deadlockCons[compute_idx])

        return -reward_funcs.Split_Directions_Cost(self.node[compute_idx],
                                                   order_x,
                                                   order_y,
                                                   self.deadlock_control,
                                                   self.deadlock_coef,
                                                   self.deadlockCons[compute_idx])

        # if is_random:
        #     return -reward_funcs.Split_Directions_Cost(self.random_nodes[compute_idx],
        #                                                order_x,
        #                                                order_y,
        #                                                deadlock_control=False,
        #                                                deadlock_coef=0.0,
        #                                                deadlockCons=0.0)
        # else:
        #     return -reward_funcs.Split_Directions_Cost(self.node[compute_idx],
        #                                                order_x,
        #                                                order_y,
        #                                                self.deadlock_control,
        #                                                self.deadlock_coef,
        #                                                self.deadlockCons[compute_idx])


def make_env(config):
    return CoreMapper(config)
