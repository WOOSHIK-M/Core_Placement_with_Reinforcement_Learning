import numpy as np
import random
import time
import pandas as pd
import copy
import matplotlib.pyplot as plt

import Code.Utils as utils
import Code.Reward_funcs as rewards


def switch_node(corenum, old_x, old_y):
    first_core = random.randint(0, corenum - 1)
    second_core = random.randint(0, corenum - 1)

    while first_core == second_core:
        second_core = random.randint(0, corenum - 1)

    new_x, new_y = copy.deepcopy(old_x), copy.deepcopy(old_y)
    new_x[first_core] = old_x[second_core]
    new_y[first_core] = old_y[second_core]
    new_x[second_core] = old_x[first_core]
    new_y[second_core] = old_y[first_core]

    return new_x, new_y


@ utils.logging_time
def run(config):
    utils.print_title("Simulated Annealing ... ")

    time.sleep(0.5)

    x_dim = config.params_dict['x_dim']
    y_dim = config.params_dict['y_dim']

    allnodenum = config.params_dict['allnodenum']
    node = config.params_dict['nodes'][0]
    phy_x = config.params_dict['phy_x']
    phy_y = config.params_dict['phy_y']

    deadlock_control = config.params_dict['deadlock_control']
    deadlock_coef = config.params_dict['deadlock_coef']
    deadlockCons = 0
    for node_i in node:
        if node_i.is_mul == 1:
            deadlockCons += 1

    SA_repeat_num = config.params_dict['SA_repeat_num']
    SA_init_temp_coef = config.params_dict['SA_init_temp_coef']
    SA_iter = config.params_dict['SA_iter']
    SA_gamma = config.params_dict['SA_gamma']
    SA_temp_threshold = config.params_dict['SA_temp_threshold']

    SA_init_temp_coefs = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 120, 120, 120, 120, 120, 1200, 1200, 1200, 1200, 1200,
                          12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 120, 120, 120, 120, 120, 1200, 1200, 1200, 1200, 1200]
    SA_iters = [150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150,
                150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150]
    SA_gammas = [0.98, 0.98, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
                 0.99, 0.99, 0.99, 0.98, 0.98, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
                 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    SA_temp_thresholds = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                          0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                          0.01, 0.01, 0.01, 0.01, 0.01]
    SA_repeat_num = len(SA_gammas)

    RS_list = np.linspace(0, allnodenum - 1, allnodenum, dtype=np.int)

    best_cost_record = []
    cost_record = []
    x_record = []
    y_record = []
    for epoch in range(SA_repeat_num):
        SA_init_temp_coef = SA_init_temp_coefs[epoch]
        SA_iter = SA_iters[epoch]
        SA_gamma = SA_gammas[epoch]
        SA_temp_threshold = SA_temp_thresholds[epoch]

        random.shuffle(RS_list)             # initialize
        temp = allnodenum * SA_init_temp_coef

        new_x = []
        new_y = []
        for i in RS_list:
            new_x.append(phy_x[i])
            new_y.append(phy_y[i])

        sample_overlaps = 1

        temp_best_cost = 1e24
        temp_best_x = []
        temp_best_y = []

        temp_record = []
        # cost_temp = rewards.Communication_Cost(node, new_x, new_y)
        cost_temp = rewards.Split_Directions_Cost(node, new_x, new_y, deadlock_control, deadlock_coef, deadlockCons)
        while temp > SA_temp_threshold:
            for _ in range(SA_iter):
                cost_old = cost_temp

                temp_record.append(cost_old)

                temp_x, temp_y = switch_node(allnodenum, new_x, new_y)
                # cost_new = rewards.Communication_Cost(node, temp_x, temp_y)
                cost_new = rewards.Split_Directions_Cost(node, temp_x, temp_y, deadlock_control, deadlock_coef, deadlockCons)

                delta_e = cost_new - cost_old

                if delta_e < 0:
                    new_x = temp_x
                    new_y = temp_y
                    cost_temp = cost_new
                else:
                    if np.exp(-delta_e / temp) > random.random():
                        new_x = temp_x
                        new_y = temp_y
                        cost_temp = cost_new

                # SAVE BEST PLACEMENT
                if cost_new < temp_best_cost:
                    temp_best_cost = cost_new
                    temp_best_x = new_x
                    temp_best_y = new_y

            temp *= SA_gamma

            sample_overlaps += 1
            if sample_overlaps % 100 == 0:
                print("Epoch: {}, \titer: {}, \tTemperature: {:.2f}, \tCost: {}, \tMin: {}".format(epoch + 1, sample_overlaps * SA_iter, temp, cost_temp, temp_best_cost))

        print("\n ==> Epoch (samples): {} ({}) is finished ... BEST: {} \n ".format(epoch + 1, (sample_overlaps - 1) * SA_iter, temp_best_cost))
        best_cost_record.append(temp_best_cost)
        cost_record.append(temp_record)
        x_record.append(temp_best_x)
        y_record.append(temp_best_y)

        new_placement = np.zeros([y_dim, x_dim], dtype=np.int)
        for i in range(1, 54):
            new_placement[temp_best_y[i]][temp_best_x[i]] = i + 1
        print("\n{}\n".format(new_placement))

    print("\n")
    utils.print_title("Finished ... ")

    print("BEST PLACEMENT of Simulated Annealing: {} (ave: {:.1f})\n".format(np.min(best_cost_record), np.average(best_cost_record)))

    utils.print_title("Saving new LUT ... ")
    for i in range(SA_repeat_num):
        new_p = utils.make_placement(config, x_record[i], y_record[i])
        new_lut = utils.placement_to_lut(new_p, config.params_dict['lut'], allnodenum, config.params_dict['lut_mul'])
        np.savetxt(config.params_dict['new_lut_path'] + "\\{}_new_lut_{}.dat".format(i, int(best_cost_record[i])), new_lut)

        cost_record_p = cost_record[i]
        # reward curve
        x_range = np.linspace(1, len(cost_record_p), len(cost_record_p))
        plt.plot(x_range, cost_record_p)
        plt.savefig(config.params_dict['plot_folder_path'] + "\\{}_reward_curve_{}.png".format(i, int(best_cost_record[i])))

        # reward data
        if len(cost_record_p) > 100:
            while len(cost_record_p) % 100 != 0:
                cost_record_p.append(0)
        datas = pd.DataFrame(np.array(cost_record_p).reshape(int(len(cost_record_p) / 100), 100))
        datas.to_csv(config.params_dict['sub_folder_path'] + "\\{}_rewards_record_{}.csv".format(i, int(best_cost_record[i])), header=None, index=None)

    print("Saved ! ")
