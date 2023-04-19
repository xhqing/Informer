from Informer.utils.tools import dotdict
from Informer.exp.exp_informer import Exp_Informer

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from args_demo import args
from preprocessing_demo import prepare_data

def set_random_seed(seed, deterministic=False):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # set_random_seed(0)
    
    train_data = prepare_data("train")
    val_data = prepare_data("val")

    model_save_path = f"informer_models/{0}_{0}.pth"
    
    exp = Exp_Informer(args)
    
    print('>>>>>> training: >>>>>>')
    exp.train(train_data, val_data, model_save_path)

    print(">>>>>>>>>>> testing: >>>>>>>")
    test_data = prepare_data("test")
    exp.test(test_data)

    torch.cuda.empty_cache()
    
    print(">>>>>>>> predicting >>>>>>>>")
    pred_data = prepare_data("pred")
    model_save_path = f"informer_models/{0}_{0}.pth"
    exp = Exp_Informer(args)
    preds = exp.predict(pred_data, model_save_path)
    print(preds)









