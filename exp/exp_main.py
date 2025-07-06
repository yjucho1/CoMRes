from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Model
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'PathFormer': Model,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        print('model size:', sum(p.numel() for p in model.parameters()))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='PathFormer':
                            outputs, expert_outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)

                else:
                    if self.args.model=='PathFormer':
                        outputs, expert_outputs = self.model(batch_x)                       
                    else:
                        outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        total_num = sum(p.numel() for p in self.model.parameters())
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            labeled_loss = []
            unlabeled_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_u, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_u = batch_u.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='PathFormer':
                            outputs, expert_outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model == 'PathFormer':
                        outputs_l, expert_outputs_l = self.model(batch_x)
                        outputs_u, expert_outputs_u = self.model(batch_u)
                        
                    else:
                        outputs, _ = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs_l = outputs_l[:, -self.args.pred_len:, f_dim:]
                    outputs_u = outputs_u[:, -self.args.pred_len:, f_dim:]

                    for i in range(self.args.num_experts):
                        expert_outputs_l[i] = expert_outputs_l[i][:, -self.args.pred_len:, f_dim:]
                    for i in range(self.args.num_experts):
                        expert_outputs_u[i] = expert_outputs_u[i][:, -self.args.pred_len:, f_dim:]

                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    l_loss = criterion(outputs_l, batch_y)
                    for i in range(self.args.num_experts):
                        l_loss += criterion(expert_outputs_l[i], batch_y)
                    l_loss /= self.args.num_experts+1
                    
                    u_loss = 0
                    for i in range(self.args.num_experts):
                        u_loss += criterion(expert_outputs_u[i], outputs_u)
                    u_loss /= self.args.num_experts
                    
                    w = 0
                    loss = l_loss ## total loss
                    if self.args.consistency:
                        w = np.clip((epoch-self.args.closs_epoch)/self.args.train_epochs, 0, 1.0)
                        loss = l_loss + w * u_loss
                    

                    train_loss.append(loss.item())
                    labeled_loss.append(l_loss.item())
                    unlabeled_loss.append(u_loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
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

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            labeled_loss = np.average(labeled_loss)
            unlabeled_loss = np.average(unlabeled_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        preds_e = []
        trues = []
        inputx = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='PathFormer':
                            outputs, expert_outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)
                else:
                    if self.args.model == 'PathFormer':
                        outputs, expert_outputs = self.model(batch_x)
                        ensemble_outputs = torch.stack(expert_outputs, dim=-1)
                        ensemble_outputs = torch.mean(ensemble_outputs, dim=-1)
                        ensemble_outputs = torch.stack([ensemble_outputs, outputs], dim=-1)
                        ensemble_outputs = torch.mean(ensemble_outputs, dim=-1)
                    else:
                        outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                ensemble_outputs = ensemble_outputs[:, -self.args.pred_len:, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                ensemble_outputs = ensemble_outputs.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                pred_e = ensemble_outputs  # batch_y.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                preds_e.append(pred_e)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        preds_e = np.array(preds_e)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        preds_e = preds_e.reshape(-1, preds_e.shape[-2], preds_e.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        mae_e, mse_e, rmse_e, mape_e, mspe_e, rse_e, corr_e = metric(preds_e, trues)
        print('mse_e:{}, mae_e:{}, rse_e:{}'.format(mse_e, mae_e, rse_e))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('mse_e:{}, mae_e:{}, rse_e:{}'.format(mse_e, mae_e, rse_e))
        f.write('\n')
        f.write('\n')
        f.close()
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))


        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='PathFormer':
                            outputs, expert_outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)

                else:
                    if self.args.model == 'PathFormer':
                        outputs, expert_outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy() 
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        return
    
    def autoregressive_test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')

        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        multi_ar_newly_mse = []
        multi_ar_newly_mae = []
        multi_ar_all_mse = []
        multi_ar_all_mae = []
        ones_ar_newly_mse = []
        ones_ar_newly_mae = []
        ones_ar_all_mse = []
        ones_ar_all_mae = []

        hop = 1
        pred_lookback = self.args.seq_len

        self.model.eval()

        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        with torch.no_grad():
            for i in range(len(test_data.data_x)-self.args.seq_len):
                inp = torch.from_numpy(test_data.data_x[i:i+self.args.seq_len]).float().reshape(1, self.args.seq_len, -1).to(self.device)
                out, _ = self.model(inp)
                ar_out = torch.empty((1, 0, out.shape[2])).to(self.device)
                all_out = out.clone().to(self.device)

                for j in range(0, pred_lookback, hop):
                    inp = torch.cat((inp[:, hop:, :], out[:, :hop, :]), axis=1)    
                    out, _ = self.model(inp)
                    ar_out = torch.cat((ar_out, out[:, -hop:, :]), axis=1)    
                    if j+hop >= self.args.pred_len:
                        break
                j = j + hop  
                all_out = torch.cat((all_out[:, j:, :], ar_out), axis=1)
                if i+j+self.args.seq_len+self.args.pred_len < len(test_data.data_x) :
                    gt = torch.from_numpy(test_data.data_x[i+j+self.args.seq_len-1:i+j+self.args.seq_len+self.args.pred_len-1]).float().reshape(1, self.args.pred_len, -1)
                    gt = gt.to(self.device) 
                    newly_mse = mse_criterion(ar_out, gt[:, -j:, :])
                    newly_mae = mae_criterion(ar_out, gt[:, -j:, :]) 
                    all_mse = mse_criterion(all_out, gt)
                    all_mae = mae_criterion(all_out, gt) 
                    multi_ar_newly_mse.append(newly_mse.item())
                    multi_ar_newly_mae.append(newly_mae.item())
                    multi_ar_all_mse.append(all_mse.item())
                    multi_ar_all_mae.append(all_mae.item())
                else:
                    break 
                if i % 100 == 0:
                    print(i)

            for i in range(len(test_data.data_x)-self.args.seq_len):
                inp = torch.from_numpy(test_data.data_x[i:i+self.args.seq_len]).float().reshape(1, self.args.seq_len, -1).to(self.device)
                out, _ = self.model(inp)
                all_out = out.clone().to(self.device)
                inp = torch.cat((inp[:, j:, :], out[:, :j, :]), axis=1)    
                out, _ = self.model(inp)
                all_out = torch.cat((all_out[:, j:, :], out[:,-j:,:]), axis=1)
                if i+j+self.args.seq_len+self.args.pred_len < len(test_data.data_x) :
                    gt = torch.from_numpy(test_data.data_x[i+j+self.args.seq_len-1:i+j+self.args.seq_len+self.args.pred_len-1]).float().reshape(1, self.args.pred_len, -1)
                    gt = gt.to(self.device)
                    newly_mse = mse_criterion(out[:, -j:, :], gt[:, -j:, :])
                    newly_mae = mae_criterion(out[:, -j:, :], gt[:, -j:, :])
                    all_mse = mse_criterion(all_out, gt)
                    all_mae = mae_criterion(all_out, gt) 
                    ones_ar_newly_mse.append(newly_mse.item())
                    ones_ar_newly_mae.append(newly_mae.item())
                    ones_ar_all_mse.append(all_mse.item())
                    ones_ar_all_mae.append(all_mae.item())

                else:
                    break
                if i % 100 == 0:
                    print(i)

        multi_ar_newly_mse = np.average(multi_ar_newly_mse)
        multi_ar_newly_mae = np.average(multi_ar_newly_mae)
        multi_ar_all_mse = np.average(multi_ar_all_mse)
        multi_ar_all_mae = np.average(multi_ar_all_mae)
        ones_ar_newly_mse = np.average(ones_ar_newly_mse)
        ones_ar_newly_mae = np.average(ones_ar_newly_mae)
        ones_ar_all_mse = np.average(ones_ar_all_mse)
        ones_ar_all_mae = np.average(ones_ar_all_mae)
        
        print('mse: ', ones_ar_newly_mse, multi_ar_newly_mse)
        print('mae: ', ones_ar_newly_mae, multi_ar_newly_mae)

        print('mse: ', ones_ar_all_mse, multi_ar_all_mse)
        print('mae: ', ones_ar_all_mae, multi_ar_all_mae)

        f = open(str(pred_lookback)+"_"+str(hop)+"_autoregressive_result.txt", 'a')
        f.write(setting + ",  ")
        f.write('{}, {}, {}, {},'.format(ones_ar_newly_mse, ones_ar_newly_mae, multi_ar_newly_mse, multi_ar_newly_mae))
        f.write('{}, {}, {}, {},'.format(ones_ar_all_mse, ones_ar_all_mae, multi_ar_all_mse, multi_ar_all_mae))
        f.write('\n')
        f.write('\n')
        f.close()
        return