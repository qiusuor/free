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
from auto_encoder import LSTMAutoEncoder, MLPAutoEncoder, weight_init, SeqMLPAutoEncoder
from collections import Counter
import platform

def load_data(train_val_split=0.7):
    data = []
    features = ["turn", "price", "open", "low", "high", "close", "pctChg"]
    
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)[features]
        data_i = [df]
        for i in range(1, K):
            data_i.append(df.shift(i))
        data_i = data_i[::-1]
        data_i = pd.concat(data_i, axis=1)
        data_i = data_i.iloc[K:]
        data.append(data_i)
        
    df = pd.concat(data)
    df = df.fillna(0).astype("float32").values
    np.random.shuffle(df)
    data = torch.from_numpy(df)
    
    N = len(data)
    # data = data.reshape(N, K, len(features))
    mean = data.mean(0)
    std = data.std(0)
    data = (data - mean) / (std + 1e-9)
    joblib.dump((mean, std), "embedding/checkpoint/mean_std_{}_{}.pkl".format(K, LAT_SIZE))
    x_train = data[:int(N*train_val_split)]
    x_test = data[int(N*train_val_split):]    
    
    return x_train, x_test
 
def parse_args():
    parser = argparse.ArgumentParser(description='MLP embedding')
    parser.add_argument('--last_n_day', type=int, default=3)
    parser.add_argument('--lat_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=500)
    
    args = parser.parse_args()
    return args   

def train(args):
    batch_size = args.batch_size
    epochs = args.epoch
    LAT_SIZE = args.lat_size
    K = args.last_n_day

    best_score = float("inf")
    best_model_path = "embedding/checkpoint/mlp_autoencoder_{}_{}.pth".format(K, LAT_SIZE)
    make_dir(best_model_path)
    torch.set_default_dtype(torch.float32)
    device = torch.device("mps") if platform.machine() == 'arm64' else torch.device("cuda")
    print("training on device: ", device)

    x_train, x_test = load_data()
    feature_dim = x_train.shape[1]
    
    criterion = nn.MSELoss(reduction="mean")
    train_loader = DataLoader(x_train, batch_size=batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(x_test, batch_size=batch_size, drop_last=True)
    model = MLPAutoEncoder(input_size=feature_dim, lat_size=LAT_SIZE)
    model.apply(weight_init)
    model.to(device)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable_params}')

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.0001,
                                betas=[0.9, 0.999],
                                weight_decay = 0.0,
                                amsgrad=False)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.3)
    mean, std = joblib.load("embedding/checkpoint/mean_std_{}_{}.pkl".format(K, LAT_SIZE))
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
        print()
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
                    print(list(zip((inputs_v.cpu()[-1,:]*(std+1e-9)+mean).numpy().tolist(), (o.cpu()[-1,:]*(std+1e-9)+mean).numpy().tolist()))[-feature_dim//K:])
            avg_vloss = np.mean(vloss_)
            print("\nTrain avg loss: {} Val avg loss: {}".format(avg_loss, avg_vloss))
            if avg_vloss < best_score:
                best_score = avg_vloss
                model_path = best_model_path
                print('    ---> New Best Score: {}. Saving model to {}'.format(best_score, model_path))
                torch.save(model.state_dict(), model_path)

                
                
    
if __name__ == "__main__":
    args = parse_args()
    train(args)
    