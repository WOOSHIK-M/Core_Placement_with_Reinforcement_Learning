import torch


class ConfigClass:
    def __init__(self):

        self.use_cuda = False
        self.device = torch.device("cpu")

        # self.use_cuda = torch.cuda.is_available()
        # self.device = torch.device("cuda:0")

        self.params_dict = {
            'ProjectFolderPath': '',
            'ConfigFilePath': '',

            # RESULT FOLDER PATH
            'rst_path': '',
            'new_lut_path': '',
            'sub_folder_path': '',
            'plot_folder_path': '',

            # MODE
            'mode': 0,      # 0 - RL, 1 - SP, 2 - RS, 3 - SA

            # CONFIGURATION
            'is_mul': True,
            'lut_filename': '',
            'lut_filename_e': '',

            'lut_normal': 256,
            'lut_mul': 512,

            'print_node_info': True,

            # (mode:1) Random Search
            'RS_repeat_num': 10,

            # (mode:2) Simulated Annealing
            'SA_repeat_num': 1,
            'SA_init_temp_coef': 120,
            'SA_iter': 150,
            'SA_gamma': 0.98,
            'SA_temp_threshold': 0.01,

            # (mode:3) Reinforcement Learning
            'is_training_mode': True,
            'load_pre_trained': False,

            'DA_mode': False,
            'dropout': 0.5,
            'random_lutnum': 100,
            'max_nodenum': 200,
            'min_nodenum': 50,
            'degrees': [8, 16, 64],
            'weights': [32, 64, 128],

            'random_features': [],
            'random_nodes': [],
            'test_features': [],

            'feature_num': 64,
            'readout_num': 256,

            'init_epochs': 10,
            'stop_ratio': 0.02,
            'max_epochs': 10000000,
            'print_interval': 50,
            'save_interval': 100,

            'ppo_clip': 0.2,
            'policy_lr': 0.0005,
            'entropy_coef': 0.0,

            'memory_buffer_size': 1024,
            'batch_size': 32,
            'mini_batch_size': 16,
            'num_worker': 1,
            'ppo_epoch': 30,

            'x_range': 1.0,
            'thr_std': 1e-3,
            'r_range': 20.0,

            'deadlock_control': False,
            'deadlock_coef': 1.0,
            'weighted_distance_coef': 1.0,

            # TOPOLOGY
            'allchipnum': 1,
            'allnodenum': 1,
            'nodeperchip': 256,

            'X_DIM': 1,
            'Y_DIM': 1,
            'x_dim': 16,
            'y_dim': 16,

            # LUT
            'lutnum': 1,

            'lut_original': [],
            'lut': [],
            'nodes': [],

            'G': [],

            'mulvec': [],

            'adjmat': [],
            'adjmat_to': [],
            'adjmat_from': [],

            'degree': [],
            'degree_to': [],
            'degree_from': [],

            'ave': [],
            'ave_to': [],
            'ave_from': [],

            'num_placed': 0,
            'vec_placed': [],

            'assignedchipidx': [],

            'chipidx': [],
            'phy_x': [],  # Zigzag for each chip
            'phy_y': [],
            'OnePhyX': [],  # Consider as Single chip
            'OnePhyY': [],
        }


class Node:
    def __init__(self):
        self.connected_to_cores = []
        self.connected_to_cores_num = 0
        self.connected_to_packets = []

        self.connected_from_cores = []
        self.connected_from_cores_num = 0
        self.connected_from_packets = []
        self.is_mul = 0
        self.multi_to_cores = []
        self.to_mul = False
        self.to_mul_lst = []
        self.to_to_mul_lst = []
        self.from_mul = False
        self.from_mul_lst = []
        self.from_from_mul_lst = []
        self.x = 0
        self.y = 0
        self.is_isolate = 0
