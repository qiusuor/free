import numpy as np
import pandas as pd
from talib import abstract
import talib
from multiprocessing import Pool
from config import *
from utils import *
from tqdm import tqdm
from joblib import dump
import warnings
from sklearn.utils import class_weight
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import platform
from sklearn.metrics import mean_squared_error, roc_curve, auc, average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

batch_size = 256
train_val_split = 0.7
epochs = 2000
label = "y_02_107"

def generate_nn_train_val():
    train_data, val_data = [], []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)
        df = train_val_data_filter(df)
        data = train_data if np.random.random() < train_val_split else val_data
        data.append(df.iloc[-300:])
    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)
    train_data = train_data.sample(frac=1).reset_index(drop=True).fillna(0)
    val_data = val_data.sample(frac=1).reset_index(drop=True).fillna(0)
    
    return train_data, val_data
    

def train_val_data_filter(df):
    return df[((df.close / df.close.shift(1)) >= 1.09) & (df.low.shift(-1) != df.high.shift(-1))]

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=256, n_classes=2):
        super().__init__()
        self.input_size = input_size
        
        self.classifer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, n_classes),
        )
        
        

    def forward(self, x):
        out = self.classifer(x)
        return F.softmax(out)
  
    
def train():
    best_ap = 0
    best_model_path = "rank/models/checkpoint/mlp_ranker.pth"
    torch.set_default_dtype(torch.float32)
    device = torch.device("mps") if platform.machine() == 'arm64' else torch.device("cuda")
    print("training on device: ", device)
    
    train_data, val_data = generate_nn_train_val()
    
    train_data = train_data[get_feature_cols() + [label]].values
    val_data = val_data[get_feature_cols() + [label]].values
    
    mean, std = torch.from_numpy(np.mean(train_data[:,:-1], axis=0)).to(device).float(), torch.from_numpy(np.std(train_data[:,:-1], axis=0)).to(device).float()
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    
    
    criterion = nn.CrossEntropyLoss()

    model = MLP(267)
    model.to(device)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable_params}')

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.0001,
                                betas=[0.9, 0.999],
                                weight_decay = 0.0,
                                amsgrad=False)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.1)


    for epoch in range(epochs):
        print('EPOCH {} / {}:'.format(epoch + 1, epochs))
        model.train()
        loss_ = []
        for i, data in enumerate(train_loader):
            data = data.float()
            inputs, targets = data[:,:-1], data[:,-1]
            
            inputs = torch.FloatTensor(inputs).to(device)
            targets = torch.LongTensor(targets.long()).to(device)
            inputs = (inputs - mean)/(std + 1e-9)
            
            yhat = model(inputs)

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

            for i, data in enumerate(val_loader):
                data = data.float()
                
                inputs_v, targets_v = data[:,:-1], data[:,-1]
                inputs_v = torch.FloatTensor(inputs_v).to(device)
                targets_v = torch.LongTensor(targets_v.long()).to(device)
                inputs_v = (inputs_v - mean)/(std + 1e-9)
                yhat_v = model(inputs_v)

                vloss = criterion(yhat_v, targets_v)
                vloss_.append(vloss.data.item())

                yhat_v_.append(yhat_v[:,1].cpu().numpy())
                targets_v_.append(targets_v.cpu().numpy())
                

            yhat_v_ = np.concatenate(yhat_v_)
            targets_v_ = np.concatenate(targets_v_)
            avg_loss_v = np.mean(vloss_)

            # Print scores

            ap = average_precision_score(targets_v_, yhat_v_)
            auc = roc_auc_score(targets_v_, yhat_v_)
            print("Train loss {} Val loss {} Val AP {} Val AUC {}".format(avg_loss, avg_loss_v, ap, auc))
            # Save best model
            # if ap > best_ap:
            #     best_ap = ap
            #     model_path = best_model_path
            #     print('    ---> New Best Score: {}. Saving model to {}'.format(auc, model_path))
            #     torch.save(model.state_dict(), model_path)

                
                
        
        
if __name__ == "__main__":
    train()
    