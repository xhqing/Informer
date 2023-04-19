from Informer.utils.tools import dotdict
from Informer.exp.exp_informer import Exp_Informer as Exp
from Informer.utils.tools import StandardScaler
from Informer.utils.timefeatures import time_features

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

def create_dataset(sequence, look_back, look_forward, task: str="MS"):

    

    data, target = [], []
    for i in range(len(sequence)-look_back-look_forward):
        data.append(sequence[i:i+look_back])
        if task in ["SS","MM"]:
            target.append(sequence[i+look_back:i+look_back+look_forward,:])
        elif task in ["MS"]:
            target.append(sequence[i+look_back:i+look_back+look_forward,-1])
    
    return np.array(data), np.array(target)



if __name__ == "__main__":
    # set_random_seed(0)

    look_back = args.seq_len
    look_forward = args.pred_len

    df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))

    cols = list(df_raw.columns); cols.remove(args.target); cols.remove('date')
    df_raw = df_raw[['date']+cols+[args.target]]

    num_train = int(len(df_raw)*0.7)
    num_test = int(len(df_raw)*0.2)
    num_vali = len(df_raw) - num_train - num_test

    scaler = StandardScaler()

    scaler.fit(seq)
    joblib.dump(scaler, f's_scaler.pkl')

    scaler = joblib.load(f's_scaler.pkl')
    seq = scaler.transform(seq)

    X, y = create_dataset(seq, look_back, look_forward)

    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(look_forward))
    model.compile(loss='mse', optimizer='adam')

    model.fit(X, y, epochs=5, batch_size=32)

    best_model_path = f"informer_models/{0}_{0}.pth"
    
    exp = Exp(args)
    
    print('>>>>>> start training: >>>>>>')
    exp.train(best_model_path)

    print('>>>>>> testing: >>>>>>')
    exp.test()

    torch.cuda.empty_cache()









