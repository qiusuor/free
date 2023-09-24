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
from auto_encoder import LSTMAutoEncoder, MLPAutoEncoder, weight_init
from collections import Counter
import platform

batch_size = 1024
epochs = 300



def load_data(train_val_split=0.7):
    data = []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)
        data.append(df)
    df = pd.concat(data)
    features = ["open", "close", "low", "high", "amount", "turn", "price", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
    df = df[features]
    df = df.fillna(0).astype("float32")
    mean = df.mean(axis=0).values
    std = df.std(axis=0).values
    df = (df - mean) / (std + 1e-9)
    joblib.dump((mean, std), os.path.join(DATA_DIR, "market", "mean_std.pkl"))
    print(mean)
    print(std)
    data = df.values
    np.random.shuffle(data)
    N = len(data)
    x_train = data[:int(N*train_val_split)]
    x_test = data[int(N*train_val_split):]
    # print(len(x_train))
    # print(x_train)
    
    
    return x_train, x_test
        

if __name__ == "__main__":
 
    best_score = 0
    best_model_path = "rank/models/checkpoint/mlp_autoencoder.pth"

    torch.set_default_dtype(torch.float32)
    device = torch.device("mps") if platform.machine() == 'arm64' else torch.device("cuda")
    print("training on device: ", device)

    x_train, x_test = load_data()
    feature_size = len(x_train[0])
    
    criterion = nn.MSELoss(reduction="mean")
    train_loader = DataLoader(x_train, batch_size=batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(x_test, batch_size=batch_size, drop_last=True)
    model = MLPAutoEncoder(input_size=feature_size, lat_size=64, hidden_size=128)
    model.apply(weight_init)
    model.to(device)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable_params}')

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.001,
                                betas=[0.9, 0.999],
                                weight_decay = 0.0,
                                amsgrad=False)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200], gamma=0.1)
    mean, std = joblib.load(os.path.join(DATA_DIR, "market", "mean_std.pkl"))
    for epoch in range(epochs):
        print('EPOCH {} / {}:'.format(epoch + 1, epochs))
        model.train()
        loss_ = []
        for i, inputs in enumerate(train_loader):
            inputs = torch.FloatTensor(inputs).to(device)
            o, lat = model(inputs)
            loss = criterion(inputs, o)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_.append(loss.data.item())
            print('\r    BATCH {} / {} loss: {}'.format(i + 1, len(train_loader), loss.data.item()), end="")
            
        scheduler.step()
        avg_loss = np.mean(loss_)

        with torch.no_grad():
            model.eval()
            vloss_ = []
            yhat_v_ = []
            targets_v_ = []
            raw_y_hat_v_ = []

            for j, inputs_v in enumerate(test_loader):

                inputs_v = torch.FloatTensor(inputs_v).to(device)
                o, lat = model(inputs_v)

                vloss = criterion(inputs_v, o)
                vloss_.append(vloss.data.item())
                
                if j%200==0:
                    print(list(zip((inputs_v.cpu()[0]*(std+1e-9)+mean).numpy().tolist(), (o.cpu()[0]*(std+1e-9)+mean).numpy().tolist())))
            avg_vloss = np.mean(vloss_)
            print("\nTrain avg loss: {} Val avg loss: {}".format(avg_loss, avg_vloss))
            if avg_vloss < best_score:
                best_score = avg_vloss
                model_path = best_model_path
                print('    ---> New Best Score: {}. Saving model to {}'.format(best_score, model_path))
                torch.save(model.state_dict(), model_path)

                
                