import numpy as np
import random
import tqdm
import time
import pandas as pd

import Code.Utils as utils
import Code.Reward_funcs as rewards


@ utils.logging_time
def run(config):
    utils.print_title("Random Search ... ")

    time.sleep(0.5)

    repeat_num = config.params_dict['RS_repeat_num']

    allnodenum = config.params_dict['allnodenum']
    node = config.params_dict['nodes']
    phy_x = config.params_dict['phy_x']
    phy_y = config.params_dict['phy_y']

    RS_list = np.linspace(0, allnodenum - 1, allnodenum, dtype=np.int)

    best_cost = 1e24
    best_x = []
    best_y = []

    cost_record = []
    for _ in tqdm.tqdm(range(repeat_num)):
        random.shuffle(RS_list)

        new_x = []
        new_y = []

        for i in RS_list:
            new_x.append(phy_x[i])
            new_y.append(phy_y[i])

        cur_rewards = rewards.Communication_Cost(node, new_x, new_y)
        cost_record.append(cur_rewards)

        if cur_rewards < best_cost:
            best_cost = cur_rewards
            best_x = new_x
            best_y = new_y

    print("\n")
    utils.print_title("Finished ... ")

    print("BEST PLACEMENT of Random Search: {} (ave: {})\n".format(best_cost, np.average(cost_record)))
    new_p = utils.make_placement(config, best_x, best_y)

    utils.print_title("Saving new LUT ... ")

    new_lut = utils.placement_to_lut(new_p, config.params_dict['lut'], allnodenum, config.params_dict['lut_mul'])

    np.savetxt(config.params_dict['new_lut_path'] + "\\new_lut_{}.dat".format(config.params_dict['RS_repeat_num']), new_lut)

    if len(cost_record) > 100:
        while len(cost_record) % 100 != 0:
            cost_record.append(0)
    datas = pd.DataFrame(np.array(cost_record).reshape(int(len(cost_record) / 100), 100))
    datas.to_csv(config.params_dict['sub_folder_path'] + "\\rewards_record.csv", header=None, index=None)

    print("Saved ! ")
