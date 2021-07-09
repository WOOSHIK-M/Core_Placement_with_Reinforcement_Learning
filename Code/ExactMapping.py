import time
import numpy as np
import pandas as pd

import Code.Utils as utils
import Code.Reward_funcs as rewards


def zigzag(node, phy_x, phy_y):
    print("zigzag ...", end=' ')
    return rewards.Split_Directions_Cost(node, phy_x, phy_y), phy_x, phy_y


def neighbor(node, phy_x, phy_y, x_dim):
    print("neighbor ...", end=' ')
    phy_x_neighbor = np.where(np.array(phy_y) % 2 == 0, np.array(phy_x), x_dim - np.array(phy_x) - 1)
    return rewards.Split_Directions_Cost(node, phy_x_neighbor, phy_y), phy_x_neighbor, phy_y


@ utils.logging_time
def run(config):
    utils.print_title("Exact Mapping ... ")

    time.sleep(0.5)

    lut_file = config.params_dict['lut_filename_e']
    allnodenum = config.params_dict['allnodenum']
    x_dim = config.params_dict['x_dim']
    node_all = config.params_dict['nodes']
    phy_x = config.params_dict['phy_x']
    phy_y = config.params_dict['phy_y']

    for idx, node in enumerate(node_all):
        print('\n')
        utils.print_title("{}".format(lut_file[idx]))
        r_zigzag, x_z, y_z = zigzag(node, phy_x, phy_y)
        print("{} ".format(r_zigzag))

        r_neighbor, x_n, y_n = neighbor(node, phy_x, phy_y, x_dim)
        print("{} ".format(r_neighbor))

    exit()

    cost_record = [['zigzag', r_zigzag], ['neighbor', r_neighbor]]
    datas = pd.DataFrame(np.array(cost_record))
    datas.to_csv(config.params_dict['sub_folder_path'] + "\\rewards_record.csv", header=None, index=None)

    new_p_z = utils.make_placement(config, x_z, y_z)
    new_lut_z = utils.placement_to_lut(new_p_z, config.params_dict['lut'], allnodenum, config.params_dict['lut_mul'])
    np.savetxt(config.params_dict['new_lut_path'] + "\\new_lut_zigzag.dat", new_lut_z)

    new_p_n = utils.make_placement(config, x_n, y_n)
    new_lut_n = utils.placement_to_lut(new_p_n, config.params_dict['lut'], allnodenum, config.params_dict['lut_mul'])
    np.savetxt(config.params_dict['new_lut_path'] + "\\new_lut_neighbor.dat", new_lut_n)

    print("Saved ! ")
