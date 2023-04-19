from args_demo import args

from Informer.utils.tools import StandardScaler
from Informer.utils.timefeatures import time_features

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

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
    
    def __init__(self):

        timeenc = 0 if args.embed!='timeF' else 1
        self.timeenc = timeenc

        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
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
            self.scaler.fit(df_data.values)
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
        dataset = PredDataset()

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
        data = (batch_x, batch_y, batch_x_mark, batch_y_mark)

    return data
