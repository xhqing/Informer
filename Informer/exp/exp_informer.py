from Informer.dataset import Dataset_Custom
from Informer.exp.exp_basic import Exp_Basic
from Informer.models.model import Informer, InformerStack
from Informer.utils.tools import EarlyStopping, adjust_learning_rate
from Informer.utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_dataloader(self, data, task: str):
        """
        task in ['train', 'val', 'test']
        """
        args = self.args

        Data = Dataset_Custom

        if task == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif task in ['train','val']:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        
        data_set = Data(data)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_loader, val_dataset, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(val_dataset, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, train_dataset, val_dataset, model_save_path):

        dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
            train_data = (batch_x, batch_y, batch_x_mark, batch_y_mark)

        dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
            val_data = (batch_x, batch_y, batch_x_mark, batch_y_mark)

        train_loader = self._get_dataloader(train_data, "train")
        vali_loader = self._get_dataloader(val_data, "val")

        path = "/".join(model_save_path.split("/")[:-1])
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_dataset, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | mse loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, val_dataset, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss(mse): {2:.7f} Vali Loss(mse): {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        self.model.load_state_dict(torch.load(model_save_path))
        
        return self.model

    def test(self, test_dataset):
        """test test_data """

        dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
            test_data = (batch_x, batch_y, batch_x_mark, batch_y_mark)

        test_loader = self._get_dataloader(test_data, 'test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(test_dataset, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse: {mse}, mae: {mae}, rmse: {rmse}, mape: {mape}, mspe: {mspe}')

    def predict(self, pred_dataset, model_save_path):
        
        self.model.load_state_dict(torch.load(model_save_path))

        self.model.eval()
        
        preds = []

        dataloader = DataLoader(pred_dataset, batch_size=len(pred_dataset), shuffle=False)
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
            pred_data = (batch_x, batch_y, batch_x_mark, batch_y_mark)

        seq_x, seq_y, seq_x_mark, seq_y_mark = pred_data[0], pred_data[1], pred_data[2], pred_data[3]
        pred, _ = self._process_one_batch(pred_dataset, seq_x, seq_y, seq_x_mark, seq_y_mark)
        
        preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        return preds

    def _process_one_batch(self, dataset, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = dataset.scaler.inverse_transform(outputs)
        
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
