from __future__ import division
import argparse
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils import class_weight
import pickle
from config import *
import pandas as pd
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from auto_encoder import LSTMAutoEncoder
from collections import Counter

batch_size = 128
feature_size = 6
epochs = 20

# def sigmoid(x):
#     s = 1 / (1 + np.exp(-x))
#     return s

def load_data(k=5, train_val_split=0.7):
    data = np.load(KMER_RAR.format(k))["arr_0"]
    # print(data)
    # print(data.keys())
    np.random.shuffle(data)
    N = len(data)
    x_train, x_test = torch.from_numpy(data[:int(N*train_val_split),:,:]), torch.from_numpy(data[int(N*train_val_split):,:,:])
    x_train = x_train.transpose(1, 2).float()
    x_test = x_test.transpose(1, 2).float()
    
    print("train samples: {}".format(len(x_train)))
    print("test samples: {}".format(len(x_test)))
    
    return x_train, x_test
        

if __name__ == "__main__":
 
    best_score = 0
    best_model_path = "rank/models/checkpoint/lstm_autoencoder.pth"

    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda")
    print("training on device: ", device)

    x_train, x_test = load_data()
    criterion = nn.MSELoss()
    x_train = DataLoader(x_train, batch_size=batch_size, drop_last=True)
    x_test = DataLoader(x_test, batch_size=batch_size, drop_last=True)

    model = LSTMAutoEncoder(input_size=feature_size, output_size=feature_size, lat_size=16)
    model.to(device)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable_params}')

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.05,
                                betas=[0.9, 0.999],
                                weight_decay = 0.0,
                                amsgrad=False)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.1)

    num_data = len(x_train)

    for epoch in range(epochs):
        print('EPOCH {} / {}:'.format(epoch + 1, epochs))
        # Training
        model.train()
        loss_ = []
        for i, inputs in enumerate(x_train):
            inputs = torch.FloatTensor(inputs).to(device)
            # targets = torch.LongTensor(targets).to(device)
            o, lat = model(inputs, device)
            # print(inputs.shape, yhat.shape, targets.shape)

            loss = criterion(inputs, o)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_.append(loss.data.item())
            print('\r    BATCH {} / {} loss: {}'.format(i + 1, num_data, loss.data.item()), end="")
            
        scheduler.step()
        avg_loss = np.mean(loss_)

        # Validation
        with torch.no_grad():
            model.eval()
            vloss_ = []
            yhat_v_ = []
            targets_v_ = []
            raw_y_hat_v_ = []

            for _, inputs_v in enumerate(x_test):

                inputs_v = torch.FloatTensor(inputs_v).to(device)
                # targets_v = torch.LongTensor(targets_v).to(device)
                o, lat = model(inputs_v, device)

                # print(inputs_v.shape, yhat_v.shape, targets_v.shape)
                vloss = criterion(inputs_v, o)
                vloss_.append(vloss.data.item())

                # yhat_v = torch.softmax(yhat_v, -1)
                # raw_y_hat_v_.append(yhat_v.cpu())
                
                # yhat_v = torch.argmax(yhat_v, dim=1)

                # targets_v_ = np.concatenate((targets_v_, targets_v.cpu()), axis=0)
                # yhat_v_ = np.concatenate((yhat_v_, yhat_v.cpu()), axis=0)
            # raw_y_hat_v_ = np.concatenate(raw_y_hat_v_)

            # Print scores
            avg_vloss = np.mean(vloss_)
            print("\nTrain avg loss: {} Val avg loss: {}".format(avg_loss, avg_vloss))
            # avg_vacc = accuracy_score(targets_v_, yhat_v_)
            # avg_vbacc = balanced_accuracy_score(targets_v_, yhat_v_)
            # avg_vf1 = f1_score(targets_v_, yhat_v_, average='macro')
            # auc = roc_auc_score(targets_v_, raw_y_hat_v_, multi_class="ovr")
            # ap = average_precision_score(targets_v_, raw_y_hat_v_, pos_label=2)
            # print()
            # print("gt 2: ", yhat_v_[targets_v_==2])
            # print("gt 0: ", yhat_v_[targets_v_==0])
            # print("gt 1: ", yhat_v_[targets_v_==1])
            
            # pos_label_score = raw_y_hat_v_[:,2]
            # pos_label_rank = np.argsort(pos_label_score)[::-1]
            # print("top 20 pos pred: ", targets_v_[pos_label_rank][:20])
            
            # print('  lr: {} Train Loss: {} Valid Loss: {} acc: {} bacc: {} f1: {} auc: {}'
            #     .format(optimizer.param_groups[0]['lr'], avg_loss,
            #             avg_vloss, avg_vacc, avg_vbacc, avg_vf1, auc))

            # Save best model
            if avg_vloss > best_score:
                best_score = avg_vloss
                model_path = best_model_path
                print('    ---> New Best Score: {}. Saving model to {}'.format(best_score, model_path))
                torch.save(model.state_dict(), model_path)

                
                