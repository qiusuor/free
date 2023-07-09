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

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def load_data(train_split=0.7, train_day_end=20220501, test_day_start=20220801):
    x_train, y_train, x_test, y_test = [], [], [], []
    with open(second_wave_retr_file, 'rb') as in_f:
        data = pickle.load(in_f)
        np.random.shuffle(data)
        data_des = pd.read_csv(second_wave_retr_des_file)
        codes = list(set(data_des.code.values))
        np.random.shuffle(codes)
        train_code = set(codes[:int(len(codes)*train_split)])
        test_code = set(codes[int(len(codes)*train_split):])
        for code, date, feat, label in data:
            # if code in train_code and label != -1:
            if code in train_code and date <= to_date(train_day_end):
                assert label != -1
                x_train.append(feat)
                y_train.append(label)
            # if code in test_code and label != -1:
            if code in test_code and date > to_date(test_day_start) and label != -1:
                x_test.append(feat)
                y_test.append(label)
    from collections import Counter
    
    print("train samples: {}".format(Counter(y_train)))
    print("test samples: {}".format(Counter(y_test)))
    
    return np.array(x_train).astype(np.float32).transpose((0, 2, 1)), y_train, np.array(x_test).astype(np.float32).transpose((0, 2, 1)), y_test          
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--feature_size',
                        action='store',
                        type=int,
                        default=4,
                        help='feature_size')
    parser.add_argument('--batch_size',
                        action='store',
                        default=256,
                        type=int,
                        help='batch size')
    parser.add_argument('--epochs',
                        action='store',
                        default=2000,
                        type=int,
                        help='number of epochs')
    parser.add_argument('--timesteps',
                        action='store',
                        default=second_wave_feature_len,
                        type=int,
                        help='number of timesteps')
   
    args = parser.parse_args()

    feature_size = args.feature_size
    timesteps = args.timesteps

    # Params
    batch_size = args.batch_size
    epochs = args.epochs

    best_score = 0
    best_model_path = "rank/models/checkpoint/tcn.pth"

    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda")
    print("training on device: ", device)

    x_train, y_train, x_test, y_test = load_data()
    num_outputs = 3
    class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train), y=y_train)
    print("class weight:", list(class_weights))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=device))
    # print(x_train.shape)
    x_train = DataLoader(x_train, batch_size=batch_size, drop_last=True)
    y_train = DataLoader(y_train, batch_size=batch_size, drop_last=True)
    x_test = DataLoader(x_test, batch_size=batch_size//4, drop_last=True)
    y_test = DataLoader(y_test, batch_size=batch_size//4, drop_last=True)

    # model = TCN(feature_size, num_outputs=num_outputs, return_sequences=False)
    model = LSTMAutoEncoder(input_size=feature_size, output_size=num_outputs, lat_size=64)
    model.to(device)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable_params}')

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.0005,
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
        for i, (inputs, targets) in enumerate(zip(x_train, y_train)):
            print('\r    BATCH {} / {}'.format(i + 1, num_data), end="")
            inputs = torch.FloatTensor(inputs).to(device)
            targets = torch.LongTensor(targets).to(device)
            o, yhat, lat = model(inputs)
            # print(inputs.shape, yhat.shape, targets.shape)

            loss = criterion(yhat, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_.append(loss.data.item())
        scheduler.step()
        avg_loss = np.mean(loss_)

        # Validation
        with torch.no_grad():
            model.eval()
            vloss_ = []
            yhat_v_ = []
            targets_v_ = []
            raw_y_hat_v_ = []

            for _, (inputs_v, targets_v) in enumerate(zip(x_test, y_test)):

                inputs_v = torch.FloatTensor(inputs_v).to(device)
                targets_v = torch.LongTensor(targets_v).to(device)
                o, yhat_v, lat = model(inputs_v)

                print(inputs_v.shape, yhat_v.shape, targets_v.shape)
                vloss = criterion(yhat_v, targets_v)
                vloss_.append(vloss.data.item())

                yhat_v = torch.softmax(yhat_v, -1)
                raw_y_hat_v_.append(yhat_v.cpu())
                
                yhat_v = torch.argmax(yhat_v, dim=1)

                targets_v_ = np.concatenate((targets_v_, targets_v.cpu()), axis=0)
                yhat_v_ = np.concatenate((yhat_v_, yhat_v.cpu()), axis=0)
            raw_y_hat_v_ = np.concatenate(raw_y_hat_v_)

            # Print scores
            avg_vloss = np.mean(vloss_)
            avg_vacc = accuracy_score(targets_v_, yhat_v_)
            avg_vbacc = balanced_accuracy_score(targets_v_, yhat_v_)
            avg_vf1 = f1_score(targets_v_, yhat_v_, average='macro')
            auc = roc_auc_score(targets_v_, raw_y_hat_v_, multi_class="ovr")
            # ap = average_precision_score(targets_v_, raw_y_hat_v_, pos_label=2)
            print()
            print("gt 2: ", yhat_v_[targets_v_==2])
            print("gt 0: ", yhat_v_[targets_v_==0])
            print("gt 1: ", yhat_v_[targets_v_==1])
            
            pos_label_score = raw_y_hat_v_[:,2]
            pos_label_rank = np.argsort(pos_label_score)[::-1]
            print("top 20 pos pred: ", targets_v_[pos_label_rank][:20])
            
            print('  lr: {} Train Loss: {} Valid Loss: {} acc: {} bacc: {} f1: {} auc: {}'
                .format(optimizer.param_groups[0]['lr'], avg_loss,
                        avg_vloss, avg_vacc, avg_vbacc, avg_vf1, auc))

            # Save best model
            if auc > best_score:
                best_score = auc
                model_path = best_model_path
                print('    ---> New Best Score: {}. Saving model to {}'.format(auc, model_path))
                torch.save(model.state_dict(), model_path)

                
                