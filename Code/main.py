import os
import time
import numpy as np
import random
import multiprocessing as mp

import Code.ParseData as parser
import ExactMapping
import RandomSearch
import SimulatedAnnealing
import ReinforcementLearning


def running(ConfigData):
    if ConfigData.params_dict['mode'] == 0:
        ExactMapping.run(ConfigData)
    elif ConfigData.params_dict['mode'] == 1:
        RandomSearch.run(ConfigData)
    elif ConfigData.params_dict['mode'] == 2:
        SimulatedAnnealing.run(ConfigData)
    elif ConfigData.params_dict['mode'] == 3:
        ReinforcementLearning.run(ConfigData)


def SingleStep():
    feature_num = [64]
    ppo_clip = [0.5]
    policy_lr = [0.005, 0.01]
    memory_buffer_size = [4, 8, 16]
    batch_size = [16, 32, 64]
    mini_batch_size = [4, 8]
    ppo_epoch = [3, 5]
    x_range = [0.08]
    r_range = [1, 5, 10, 30]
    thr_std = [0.0008]

    while True:
        time.sleep(5)

        ConfigData = parser.make_config()

        random.shuffle(feature_num)
        random.shuffle(ppo_clip)
        random.shuffle(policy_lr)
        random.shuffle(memory_buffer_size)
        random.shuffle(batch_size)
        random.shuffle(mini_batch_size)
        random.shuffle(ppo_epoch)
        random.shuffle(x_range)
        random.shuffle(r_range)
        random.shuffle(thr_std)

        ConfigData.params_dict['feature_num'] = feature_num[0]
        ConfigData.params_dict['ppo_clip'] = ppo_clip[0]
        ConfigData.params_dict['policy_lr'] = policy_lr[0]
        ConfigData.params_dict['memory_buffer_size'] = memory_buffer_size[0]
        ConfigData.params_dict['batch_size'] = batch_size[0]
        ConfigData.params_dict['mini_batch_size'] = mini_batch_size[0]
        ConfigData.params_dict['ppo_epoch'] = ppo_epoch[0]
        ConfigData.params_dict['x_range'] = x_range[0]
        ConfigData.params_dict['r_range'] = r_range[0]
        ConfigData.params_dict['thr_std'] = thr_std[0]

        running(ConfigData)

        # exit()


if __name__ == "__main__":
    # RECORD START TIME
    now = time.localtime()
    print("\n==>  Mapping Processes start at %04d/%02d/%02d %02d:%02d:%02d\n"
          % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

    SingleStep()

    # processes = [mp.Process(target=SingleStep) for _ in range(12)]
    #
    # print('start')
    # for proc in processes:
    #     proc.start()
    #     time.sleep(10)
    #
    # print('join')
    # for proc in processes:
    #     proc.join()
