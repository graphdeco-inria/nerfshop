from numpy import require
import torch
import argparse
import json

import pyngp as ng
import pyngp_bindings
from pyngp import Testbed, NeRFNetwork

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--scene_dir", type=str, required=True)

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()

    testbed = Testbed(args.scene_dir, args.model_config)

    testbed.train()
    


    # # Read network config file
    # with open(args.model_config) as handle:
    #     model_config = json.load(handle)

    # net = NeRFNetwork(model_config)