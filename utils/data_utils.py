import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def prepare_data(dataset_train, apply_scaling=True):
    X_train = torch.cat([traj[:-1] for traj in dataset_train])
    Y_train = torch.cat([traj[1:] for traj in dataset_train])

    if apply_scaling:
        scaler = MinMaxScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train.numpy()), dtype=torch.float32)
        Y_train = torch.tensor(scaler.transform(Y_train.numpy()), dtype=torch.float32)
    else:
        scaler = None

    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    return X_train, Y_train, dataloader, scaler

def inverse_min_max_scale(data, scaler):
    if scaler is not None:
        return scaler.inverse_transform(data)
    return data
