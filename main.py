import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from algorithms.ERM.src.Trainer_ERM import Trainer_ERM
from algorithms.mDSDI.src.Trainer_mDSDI import Trainer_mDSDI
from algorithms.mHSHA.src.Trainer_mHSHA import Trainer_mHSHA
from algorithms.mDSDI_ssl.src.Trainer_mDSDI_ssl import Trainer_mDSDI_ssl



import wandb


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

algorithms_map = {"ERM": Trainer_ERM, "mDSDI": Trainer_mDSDI, "mHSHA": Trainer_mHSHA, "mDSDI_ssl": Trainer_mDSDI_ssl}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default='./algorithms/mDSDI/configs/synthetic_theta5_100.json', help="Path to configuration file")
    parser.add_argument("--exp_idx", default='1',  help="Index of experiment")
    parser.add_argument("--gpu_idx", default='1', help="Index of GPU")
    bash_args = parser.parse_args()
    with open(bash_args.config, "r") as inp:
        args = argparse.Namespace(**json.load(inp))
    

    now = datetime.now().isoformat(timespec='minutes')
    project_name = "mHSHA All experiments"
    wandb.config = {'batch size': args.batch_size, 'learning_rate': args.learning_rate}
    wandb.init(config = wandb.config, project= project_name, entity="msikarou", name=str(args.exp_name + '/' + args.algorithm + now))
    os.environ["CUDA_VISIBLE_DEVICES"] = bash_args.gpu_idx

    fix_random_seed(args.seed_value)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = algorithms_map[args.algorithm](args, device, bash_args.exp_idx)
    trainer.train()
    trainer.test()
    trainer.save_plot()
    print("Finished!")