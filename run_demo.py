from Informer.utils.tools import dotdict
from Informer.exp.exp_informer import Exp_Informer
from Informer.utils.tools import StandardScaler
from Informer.utils.timefeatures import time_features

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch

# --------------- args --------------------
args = dotdict()
args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

args.root_path = './' # root path of data file
args.data_path = 'ETTh1.csv' # data file

args.scale = False
args.inverse = False

args.features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = 'OT' # target feature in S or MS task
args.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h

args.seq_len = 30*24           # input sequence length of Informer encoder
args.label_len = 7*24      # start token length of Informer decoder
args.pred_len = 24                # prediction sequence length
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

args.enc_in = 7   # encoder input size
args.dec_in = 7   # decoder input size
args.c_out = 1          # output size

args.factor = 5 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 2     # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0. # dropout
args.attn = 'prob' # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = True # whether to use distilling in encoder
args.output_attention = False # whether to output attention in ecoder
args.mix = True
args.padding = 0

args.batch_size = 32
args.learning_rate = 0.0001
args.loss = 'mse'
args.lradj = 'type1' # learning rate adjust
args.use_amp = False # whether to use automatic mixed precision training

args.num_workers = 0
args.train_epochs = 6
args.patience = 3

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0 # device string: cuda:0

args.use_multi_gpu = False
args.devices = '0,1,2,3'

# ------------------- data prepare --------------------

class Preprocessing(Dataset):

    def __init__(self, flag='train'):
        
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        timeenc = 0 if args.embed!='timeF' else 1
        self.timeenc = timeenc
        
        self.__read_data__()

    def __read_data__(self):

        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))

        cols = list(df_raw.columns); cols.remove(args.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[args.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train-args.seq_len, len(df_raw)-num_test-args.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if args.features=='M' or args.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif args.features=='S':
            df_data = df_raw[[args.target]]

        if args.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=args.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + args.seq_len
        r_begin = s_end - args.label_len
        r_end = r_begin + args.label_len + args.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        return len(self.data_x) - args.seq_len - args.pred_len + 1

class PredDataset(Dataset):
    
    def __init__(self, train_scaler):
        
        self.scaler = train_scaler

        timeenc = 0 if args.embed!='timeF' else 1
        self.timeenc = timeenc

        self.__read_data__()
    
    def __read_data__(self):
        
        df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))

        cols = list(df_raw.columns); cols.remove(args.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[args.target]]

        border1 = len(df_raw)-args.seq_len
        border2 = len(df_raw)

        if args.features=='M' or args.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif args.features=='S':
            df_data = df_raw[[args.target]]

        if args.scale:
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values 

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=args.pred_len+1, freq=args.freq)

        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=args.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp        


    def __len__(self):
        return len(self.data_x) - args.seq_len + 1 

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + args.seq_len
        r_begin = s_end - args.label_len
        r_end = r_begin + args.label_len + args.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin+args.label_len]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


def prepare_data(task: str) -> tuple:
    """
    task in ['train','test','val','pred']
    """
    if task in ['train','test','val']:
        dataset = Preprocessing(task)
    elif task=='pred':
        train_scaler = Preprocessing("train").scaler
        dataset = PredDataset(train_scaler)
    return dataset

# ------------------- random seed ----------------------

def set_random_seed(seed, deterministic=False):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ------------------- main ------------------

if __name__ == "__main__":
    # set_random_seed(0)
    
    train_dataset = prepare_data("train")
    val_dataset = prepare_data("val")

    model_save_path = f"informer_models/{0}_{0}.pth"
    
    exp = Exp_Informer(args)
    
    print('>>>>>> training: >>>>>>')
    exp.train(train_dataset, val_dataset, model_save_path)

    print(">>>>>>>>>>> testing: >>>>>>>")
    test_dataset = prepare_data("test")
    exp.test(test_dataset)

    torch.cuda.empty_cache()
    
    print(">>>>>>>> predicting >>>>>>>>")
    pred_dataset = prepare_data("pred")
    model_save_path = f"informer_models/{0}_{0}.pth"
    exp = Exp_Informer(args)
    preds = exp.predict(pred_dataset, model_save_path)
    print(preds)









