config = {
    "lut_info": {
        # example: toy_lut.txt
        # lut file should be located in data/
        "lut_file": "toy_lut.txt",

        "is_multicast": True,   # multicast info will be ignored when it is False
    },

    "env_config": {
        # topology info (chips(row, col), cores(row, col))
        "grid": (1, 1, 2, 5),

        "reward_config": {
            # supported reward function
            #   - Communication_cost
            "reward_method": "Communication_cost",
            "deadlock_constraint": False,
            "deadlock_coef": 0.1,
        }    
    },

    # EM - Exact mapping (zigzag, neighbor)
    # RS - Random Search
    # SA - Simulated Annealing
    # RL - Reinforcement Learning
    "mode": "RL",

    "RS_config": {
        "repeat_num": 1000,
    },

    "SA_config": {
        "init_temp_coef": 120, 
        "n_iters": 150,
        "gamma": 0.98,
        "temp_threshold": 0.1,
    },

    "RL_config": {
        "use_cuda": False,
        "device": 0,    # gpu_num if use_cuda

        "batch_size": 64,
        "ppo_epoch": 5,
        "ppo_clip": 0.1,
        "lr": 0.0005,
    }
}