import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from algorithms.ERM.src.Trainer_ERM import Trainer_ERM
from algorithms.mDSDI.src.Trainer_mDSDI import Trainer_mDSDI

import wandb
exp_name = "hospital agnostic 2_1"
test = wandb.init(config = wandb.config, project= exp_name, entity="msikarou")

def fix_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


algorithms_map = {"ERM": Trainer_ERM, "mDSDI": Trainer_mDSDI}
# def test_logger_function(logger):
#     logger.log({'Accuracy/train': 100.0})




# class test_logger_class():
#     def __init__(self):
#         # super(ZS_Domain_Classifier, self).__init__()
#         # self.logger = logger
#         wandb.log({'Accuracy/train/in class': 100.0})

# def test_wandb():
#     test.log({'test':3})

# class test_wandb_class:
#     def __init__(self):
#         pass
#     def method_test(self):
#         test.log({'test':3})



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default='./algorithms/mDSDI/configs/RCC_nci.json', help="Path to configuration file")
    parser.add_argument("--exp_idx", default='1',  help="Index of experiment")
    parser.add_argument("--gpu_idx", default='1', help="Index of GPU")
    bash_args = parser.parse_args()
    with open(bash_args.config, "r") as inp:
        args = argparse.Namespace(**json.load(inp))

    os.environ["CUDA_VISIBLE_DEVICES"] = bash_args.gpu_idx

    fix_random_seed(args.seed_value)
    logging.basicConfig(
        filename="algorithms/" + args.algorithm + "/results/logs/" + args.exp_name + "_" + bash_args.exp_idx + ".log",
        filemode="w",
        level=logging.INFO,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = algorithms_map[args.algorithm](args, device, bash_args.exp_idx)
    # trainer = Trainer_mDSDI(args, device, bash_args.exp_idx)
    trainer.train()
    trainer.test()
    trainer.save_plot()
    print("Finished!")