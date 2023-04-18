from Informer.utils.tools import dotdict
from Informer.exp.exp_informer import Exp_Informer as Exp

import torch
import numpy as np
import pandas as pd

from args_demo import args

def set_random_seed(seed, deterministic=False):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # set_random_seed(0)

    best_model_path = f"informer_models/{0}_{0}.pth"
    
    exp = Exp(args)
    
    print('>>>>>> start training: >>>>>>')
    exp.train(best_model_path)

    print('>>>>>> testing: >>>>>>')
    exp.test()

    torch.cuda.empty_cache()









