import numpy as np
import random
import time
import pandas as pd
import copy
import warnings
import torch.multiprocessing as mp

from Code.Agent import Agent
import Code.Utils as utils


# @ utils.logging_time
def run_init_episode(worker):
    # utils.print_title("Init Episodes ...")

    worker.init_network()


# @ utils.logging_time
def run_training_episode(worker):
    # utils.print_title('Start Training...')

    worker.train_network()


# @ utils.logging_time
def run(config):
    worker = Agent(config)

    # run initial episodes
    run_init_episode(worker)

    # start training
    run_training_episode(worker)

