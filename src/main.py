import argparse
import json

import utils
from parsedata import DataCollector


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="config/example.py",
    help="configuration file path",
)
args = parser.parse_args()


def runner(data: DataCollector):
    """Implement a mapping process according to the 'mode'."""
    
    mode = data.config["mode"]

    if mode == "EM":
        from runner.exact_mapping import run
    elif mode == "RS":
        from runner.random_search import run
    elif mode == "SA":
        from runner.simulated_annealing import run
    elif mode == "RL":
        from runner.reinforcement_learning import run
    else:
        raise NotImplementedError(f"mode `{mode}` is not supported ...")

    run(data)


if __name__ == "__main__":
    # parsing data
    utils.print_title("Read Configurations ...")
    data = DataCollector(args.config)

    # print parsed data
    print(f"Filename: {args.config}")
    print(json.dumps(data.config, sort_keys=False, indent=4))

    # do mapping !
    runner(data)

