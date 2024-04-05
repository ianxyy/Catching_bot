import h5py
import numpy as np
import math
import csv

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, global_max_pool, fps, radius, knn_interpolate
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
from joblib import dump, load


class GraspDataset(Dataset):
    def __init__(self, h5_file, mode = 'train'):
        super(GraspDataset, self).__init__()
        self.h5_file = h5_file
        self.mode = mode
        self.traj_scaler = StandardScaler()
        self.pc_scaler = MinMaxScaler(feature_range=(0, 1))
        # self.time_scaler = StandardScaler()
        self.X_WG1_scaler = StandardScaler()
        self.X_WG2_scaler = StandardScaler()
        if mode == 'train':
            self.traj_after_scaler = StandardScaler()
            self.compute_normalization_parameters()          

    def compute_normalization_parameters(self):
        with h5py.File(self.h5_file, 'r') as hf:
            all_traj_data = []
            all_traj_data_after = []
            all_pc_data = []
            all_time_data = []
            all_X_WG1_data = []
            all_X_WG2_data = []
            for i in range(len(hf.keys()) // 9):
                traj_data = hf[f'traj_data_{i}'][:]
                traj_data_after = hf[f'traj_data_after_{i}'][:]
                pc_data = hf[f'pc_data_{i}'][:]
                # time_data = hf[f'time_data_{i}'][:]
                X_WG1_data = hf[f'X_WG1_{i}'][:]
                X_WG2_data = hf[f'X_WG2_{i}'][:]
                all_traj_data.append(traj_data)
                all_traj_data_after.append(traj_data_after)
                all_pc_data.append(pc_data)
                print('traj_data',traj_data.shape)
                
                # all_time_data.append(time_data)
                all_X_WG1_data.append(X_WG1_data)
                all_X_WG2_data.append(X_WG2_data)
            all_traj_data = np.concatenate(all_traj_data, axis=0)
            print('all_traj_data',all_traj_data.shape)
            print('all_traj_data',all_traj_data.reshape(-1, all_traj_data.shape[-1]).shape)
            all_traj_data_after = np.concatenate(all_traj_data_after, axis=0)
            all_pc_data = np.concatenate(all_pc_data, axis=0)
            # all_time_data = np.concatenate(all_time_data, axis=0).reshape(-1, all_time_data[0].shape[-1])
            # all_X_WG1_data = np.concatenate(all_X_WG1_data, axis=0).reshape(-1, all_X_WG1_data[0].shape[-1])
            # all_X_WG2_data = np.concatenate(all_X_WG2_data, axis=0).reshape(-1, all_X_WG2_data[0].shape[-1])
            self.traj_scaler.fit(all_traj_data.reshape(-1, all_traj_data.shape[-1]))
            self.traj_after_scaler.fit(all_traj_data_after.reshape(-1, all_traj_data_after.shape[-1]))
            self.pc_scaler.fit(all_pc_data.reshape(-1, all_pc_data.shape[-1]))
            # self.time_scaler.fit(all_time_data)
            self.X_WG1_scaler.fit(all_X_WG1_data)
            self.X_WG2_scaler.fit(all_X_WG2_data)

    def len(self):
        with h5py.File(self.h5_file, 'r') as hf:
            num_types_per_sample = 7 if self.mode == 'test' else 9
            return len(hf.keys()) // num_types_per_sample

    def get(self, idx):
        with h5py.File(self.h5_file, 'r') as hf:
            if self.mode == 'test':
                # Load data
                traj_data = hf[f'traj_data_{idx}'][:]
                pc_data = hf[f'pc_data_{idx}'][:]
                time_data = hf[f'time_data_{idx}'][:]
                X_WG1_data = hf[f'X_WG1_{idx}'][:]
                X_WG2_data = hf[f'X_WG2_{idx}'][:]
                obj_catch_t_data = hf[f'obj_catch_t_{idx}'][()]
                result_data = hf[f'result_{idx}'][()]

                return {
                    'pc' : torch.tensor(pc_data, dtype=torch.float32),
                    'traj': torch.tensor(traj_data, dtype=torch.float32),             #5,150,16
                    'time': torch.tensor(time_data, dtype=torch.float32),
                    'X_WG1': torch.tensor(X_WG1_data, dtype=torch.float32),
                    'X_WG2': torch.tensor(X_WG2_data, dtype=torch.float32),
                    'obj_catch_t': torch.tensor(obj_catch_t_data, dtype=torch.float32).unsqueeze(0),
                    'result': torch.tensor(result_data, dtype=torch.float32).unsqueeze(0),
                }
            
            else:
                # Load data
                traj_data = hf[f'traj_data_{idx}'][:]
                traj_data_after = hf[f'traj_data_after_{idx}'][:]
                pc_data = hf[f'pc_data_{idx}'][:]
                time_data = hf[f'time_data_{idx}'][:]
                time_after_data = hf[f'time_data_after_{idx}'][:]
                X_WG1_data = hf[f'X_WG1_{idx}'][:]
                X_WG2_data = hf[f'X_WG2_{idx}'][:]
                obj_catch_t_data = hf[f'obj_catch_t_{idx}'][()]
                result_data = hf[f'result_{idx}'][()]
                
                # Normalize trajectory data
                traj_data_normalized = self.traj_scaler.transform(traj_data.reshape(-1, traj_data.shape[-1])).reshape(traj_data.shape)
                traj_data_after_normalized = self.traj_after_scaler.transform(traj_data_after.reshape(-1, traj_data_after.shape[-1])).reshape(traj_data_after.shape)
                # time_data_normalized = self.time_scaler.transform(time_data.reshape(1, -1)).flatten()
                # Concatenate time data to traj_data
                time_data_expanded = time_data[:, None, None]
                time_data_replicated = np.repeat(time_data_expanded, traj_data_normalized.shape[1], axis=1)
                traj_data_with_time = np.concatenate([traj_data_normalized, time_data_replicated], axis=2)
                # print('time', traj_data_with_time)#[:,1,-1])
                # print('time', time_data_replicated)
                time_data_after_expanded = time_after_data[:, None, None]
                time_data_after_replicated = np.repeat(time_data_after_expanded, traj_data_after_normalized.shape[1], axis=1)
                traj_data_after_with_time_after = np.concatenate([traj_data_after_normalized, time_data_after_replicated], axis=2)
                # Normalize point cloud data
                pc_data_normalized = self.pc_scaler.transform(pc_data.reshape(-1, pc_data.shape[-1])).reshape(pc_data.shape)

                # Standardize gripper data
                X_WG1_data_normalized = self.X_WG1_scaler.transform(X_WG1_data.reshape(-1, X_WG1_data.shape[-1])).reshape(X_WG1_data.shape)
                X_WG2_data_normalized = self.X_WG2_scaler.transform(X_WG2_data.reshape(-1, X_WG2_data.shape[-1])).reshape(X_WG2_data.shape)
                
                # Save scaler 
                dump(self.traj_scaler, 'traj_scaler.joblib')
                dump(self.traj_after_scaler, 'traj_after_scaler.joblib')
                dump(self.pc_scaler, 'pc_scaler.joblib')
                dump(self.X_WG1_scaler, 'X_WG1_scaler.joblib')
                dump(self.X_WG2_scaler, 'X_WG2_scaler.joblib')

                # Convert to tensors
                traj_with_time_tensor = torch.tensor(traj_data_with_time, dtype=torch.float32)
                traj_after_with_time_after_tensor = torch.tensor(traj_data_after_with_time_after, dtype=torch.float32)
                pc_tensor = torch.tensor(pc_data_normalized, dtype=torch.float32)
                X_WG1_tensor = torch.tensor(X_WG1_data_normalized, dtype=torch.float32)
                X_WG2_tensor = torch.tensor(X_WG2_data_normalized, dtype=torch.float32)
                obj_catch_t_tensor = torch.tensor(obj_catch_t_data, dtype=torch.float32).unsqueeze(0)
                result_tensor = torch.tensor(result_data, dtype=torch.float32).unsqueeze(0)
                # print(f'traj_data_with_time_normalized shape:{traj_with_time_tensor.shape}')
                # print(f'pc_data_normalized shape:{pc_tensor.shape}')
                # print(f'time_data shape:{time_data_normalized.shape}')
                # print(f'X_WG1 shape:{X_WG1_tensor.shape}')
                # print(f'X_WG2 shape:{X_WG2_tensor.shape}')
                # print(f'obj_catch_t_data shape:{obj_catch_t_tensor.shape}')
                # print(f'result_data shape:{result_tensor.shape}')

                pointnet_input = [Data(pos = pc_tensor[i]) for i in range(pc_tensor.size(0))]
                pointnet_batch = Batch.from_data_list(pointnet_input)
                # print('size',pointnet_batch.size())

                return {
                    'pointnet_input' : pointnet_batch,
                    'transformer_input_src': traj_with_time_tensor,              #5,150,16
                    'transformer_input_tgt': traj_after_with_time_after_tensor,  #16,150,16
                    'pc': pc_tensor,
                    'X_WG1': X_WG1_tensor,
                    'X_WG2': X_WG2_tensor,
                    'obj_catch_t': obj_catch_t_tensor,
                    'result': result_tensor,
                }


class SAModule(nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class PointNetPlusPlus(nn.Module):
    def __init__(self):
        super(PointNetPlusPlus, self).__init__()
        self.sa1_module = SAModule(0.5, 0.2, Sequential(Linear(3, 64), ReLU(), Linear(64, 64), ReLU()))
        self.sa2_module = SAModule(0.25, 0.4, Sequential(Linear(64 + 3, 128), ReLU(), Linear(128, 128), ReLU()))
        self.global_sa_module = GlobalSAModule(Sequential(Linear(128 + 3, 256), ReLU(), Linear(256, 1024)))

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        x, pos, batch = self.global_sa_module(*sa2_out)
        return x


class TrajTransformer(nn.Module):
    def __init__(self, feature_size, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
        super(TrajTransformer, self).__init__()
        self.feature_size = feature_size
        self.pos_encoder = PositionalEncoding(feature_size, max_seq_length)
        self.transformer = nn.Transformer(d_model=feature_size, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward)
        self.decoder = nn.Linear(feature_size, feature_size)

    def forward(self, src, tgt = None):
        src = self.pos_encoder(src)
        if tgt is None:
            #inference mode
            batch_size = src.size(1)
            pred_len = 16*150  # Number of steps to predict
            tgt = torch.zeros(pred_len, batch_size, src.size(2), device=src.device)
            
            # Iteratively predict future steps
            for i in range(pred_len):
                tgt_temp = self.pos_encoder(tgt)
                output = self.transformer(src, tgt_temp)
                # Update tgt with the latest prediction
                tgt[i] = output[i]
        else:
            tgt = self.pos_encoder(tgt)
            output = self.transformer(src, tgt)

        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].to(x.device)


class PredictionMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(PredictionMLP, self).__init__()
        layers = []
        for hs in hidden_sizes:
            layers.append(nn.Linear(input_size, hs))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.2))
            input_size = hs  # Next layer's input size is the current layer's output size
        
        # Final layer without ReLU to allow for negative values and more flexibility
        self.features = nn.Sequential(*layers)
        self.output_gripper1 = nn.Linear(hidden_sizes[-1], 9)  # Assuming gripper pose is a 3D vector
        self.output_gripper2 = nn.Linear(hidden_sizes[-1], 9)
        self.output_catch_time = nn.Linear(hidden_sizes[-1], 1)  # Catch time is a single scalar value

    def forward(self, x):
        features = self.features(x)
        xw_1 = self.output_gripper1(features)
        xw_2 = self.output_gripper2(features)
        obj_catch_t = self.output_catch_time(features)
        return xw_1, xw_2, obj_catch_t
    
def reshape_for_transformer(data):
    # Assuming data is of shape [batch_size, timesteps, points, features]
    batch_size, timesteps, points, features = data.size()
    # Reshape to [batch_size, timesteps * points, features]
    reshaped_data = data.view(batch_size, timesteps * points, features)
    # Transpose to match Transformer's expected input shape [seq_len, batch, features]
    transformer_ready_data = reshaped_data.transpose(0, 1)
    return transformer_ready_data

def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def train(pointnet_model, transformer_model, mlp_model, optimizer, criterion, trainloader, batch_size, log=True):
    pointnet_model.train()
    transformer_model.train()
    mlp_model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(trainloader):
        # print(batch['X_WG1'].size(0))
        if batch['X_WG1'].size(0) < batch_size:
            break
        batch['pointnet_input'] = batch['pointnet_input'].to(device)
        batch['transformer_input_src'] = batch['transformer_input_src'].to(device)
        batch['transformer_input_tgt'] = batch['transformer_input_tgt'].to(device)
        xw_1_true = batch['X_WG1'].to(device)
        xw_2_true = batch['X_WG2'].to(device)
        obj_catch_t_true = batch['obj_catch_t'].to(device)
        
        pointnet_out = pointnet_model(batch['pointnet_input'])                          #[batch_size,1024]
        # print('size',batch['pointnet_input'].size())
        src_transformer = reshape_for_transformer(batch['transformer_input_src'])       #[5x150,batch_size,16]
        tgt_transformer = reshape_for_transformer(batch['transformer_input_tgt'])       #[16x150,batch_size,16]
        transformer_out = transformer_model(src_transformer, tgt_transformer)           #[16x150,batch_size,16]
        # print('size:',pointnet_out.size())
        pointnet_out_agg = pointnet_out.view(batch_size, 5, 1024).mean(dim=1)  # Mean pooling over the 5 dimension [batch_size, 1024]
        transformer_output_agg = transformer_out.view(16, 150, batch_size, 16).mean(dim=1)  # Mean pooling over the 150 dimension [16, batch_size, 16]
        transformer_output_agg_flat = transformer_output_agg.transpose(0, 1).reshape(batch_size, -1)  # [batch_size, 16*16]
        combined_features = torch.cat((pointnet_out_agg, transformer_output_agg_flat), dim=1)  # [batch_size, 1024 + 16*16]

        optimizer.zero_grad()
        xw_1_pred, xw_2_pred, obj_catch_t_pred = mlp_model(combined_features)
        loss = criterion(xw_1_pred, xw_1_true) + criterion(xw_2_pred, xw_2_true) + criterion(obj_catch_t_pred, obj_catch_t_true)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_losses.append(train_loss/ (batch_idx + 1))
    return train_loss, train_losses

def val(pointnet_model, transformer_model, mlp_model, val_loader):
    pointnet_model.eval()
    transformer_model.eval()
    mlp_model.eval()  # Evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch['X_WG1'].size(0) < batch_size:
                break
            batch['pointnet_input'] = batch['pointnet_input'].to(device)
            batch['transformer_input_src'] = batch['transformer_input_src'].to(device)
            batch['transformer_input_tgt'] = batch['transformer_input_tgt'].to(device)
            xw_1_true = batch['X_WG1'].to(device)
            xw_2_true = batch['X_WG2'].to(device)
            obj_catch_t_true = batch['obj_catch_t'].to(device)
            
            pointnet_out = pointnet_model(batch['pointnet_input'])                          #[batch_size,1024]
            src_transformer = reshape_for_transformer(batch['transformer_input_src'])       #[5x150,batch_size,16]
            tgt_transformer = reshape_for_transformer(batch['transformer_input_tgt'])       #[16x150,batch_size,16]
            transformer_out = transformer_model(src_transformer, tgt_transformer)           #[16x150,batch_size,16]

            pointnet_out_agg = pointnet_out.view(batch_size, 5, 1024).mean(dim=1)  # Mean pooling over the 5 dimension [batch_size, 1024]
            transformer_output_agg = transformer_out.view(16, 150, batch_size, 16).mean(dim=1)  # Mean pooling over the 150 dimension [16, batch_size, 16]
            transformer_output_agg_flat = transformer_output_agg.transpose(0, 1).reshape(batch_size, -1)  # [batch_size, 16*16]
            combined_features = torch.cat((pointnet_out_agg, transformer_output_agg_flat), dim=1)  # [batch_size, 1024 + 16*16]

            xw_1_pred, xw_2_pred, obj_catch_t_pred = mlp_model(combined_features)
            loss = criterion(xw_1_pred, xw_1_true) + criterion(xw_2_pred, xw_2_true) + criterion(obj_catch_t_pred, obj_catch_t_true)
            loss = loss.mean()
            val_loss += loss.item()
    val_losses.append(val_loss/ (batch_idx + 1))
    return val_loss, val_losses
    

def test(pointnet_model, transformer_model, mlp_model):
    batch_size = 1
    test_dataset = GraspDataset('filtered_graspnet_data.h5', 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    pointnet_model.load_state_dict(torch.load('pointnet_model_weights.pth', map_location=torch.device('cuda')))
    transformer_model.load_state_dict(torch.load('transformer_model_weights.pth', map_location=torch.device('cuda')))
    mlp_model.load_state_dict(torch.load('mlp_model_weights.pth', map_location=torch.device('cuda')))

    pointnet_model.eval()
    transformer_model.eval()
    mlp_model.eval()  # Evaluation mode    

    traj_scaler_path = 'traj_scaler.joblib'
    pc_scaler_path = 'pc_scaler.joblib'
    X_WG1_scaler_path = 'X_WG1_scaler.joblib'
    X_WG2_scaler_path = 'X_WG2_scaler.joblib'

    traj_scaler = load(traj_scaler_path)
    pc_scaler = load(pc_scaler_path)
    X_WG1_scaler = load(X_WG1_scaler_path)
    X_WG2_scaler = load(X_WG2_scaler_path)

    criterion = torch.nn.MSELoss()  # Define the loss function
    total_loss = 0.0
    with open('test_predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Obj_Catch_T_True', 'Obj_Catch_T_Pred', 'X_WG1_True', 'X_WG1_Pred', 'X_WG2_True', 'X_WG2_Pred'])
        with torch.no_grad(): 
            for batch_idx, batch in enumerate(test_loader):
                traj_data = batch['traj'].numpy().squeeze() 
                pc_data =batch['pc'].numpy().squeeze() 
                time_data = batch['time'].numpy().squeeze() 
                xw_1_true = batch['X_WG1'].to('cuda')
                xw_2_true = batch['X_WG2'].to('cuda')
                obj_catch_t_true = batch['obj_catch_t'].to('cuda')
                result_data = batch['result'].to('cuda')

                traj_input_normalized = traj_scaler.transform(traj_data.reshape(-1, traj_data.shape[-1])).reshape(traj_data.shape)
                time_data_expanded = time_data[:, None, None]
                time_data_replicated = np.repeat(time_data_expanded, traj_input_normalized.shape[1], axis=1)
                # print('traj',traj_input_normalized.shape)
                # print('time',time_data_replicated.shape)
                traj_data_with_time = np.concatenate([traj_input_normalized, time_data_replicated], axis=2)
                # print(traj_data_with_time)


                pointcloud_input_normalized = pc_scaler.transform(pc_data.reshape(-1, pc_data.shape[-1])).reshape(pc_data.shape)
                
                traj_input_normalized_tensor = torch.tensor(traj_data_with_time, dtype=torch.float32).to('cuda')
                pointcloud_input_normalized_tensor = torch.tensor(pointcloud_input_normalized, dtype=torch.float32).to('cuda')
                # pointcloud_input_normalized_tensor = pointcloud_input_normalized_tensor.unsqueeze(0)  # Add batch dimension
                
                # print('shape', pc_data.shape)
                pointnet_input = [Data(pos = pointcloud_input_normalized_tensor[i]) for i in range(pointcloud_input_normalized_tensor.size(0))]
                pointnet_batch = Batch.from_data_list(pointnet_input)
                # print('batch_shape', pointnet_batch.size())

                timesteps, points, features = traj_input_normalized_tensor.size()
                # Reshape to [batch_size, timesteps * points, features]
                reshaped_data = traj_input_normalized_tensor.view(batch_size, timesteps * points, features)  
                # Transpose to match Transformer's expected input shape [seq_len, batch, features]
                src_transformer = reshaped_data.transpose(0, 1)

                pointnet_out = pointnet_model(pointnet_batch)
                transformer_out = transformer_model(src = src_transformer, tgt = None)

                pointnet_out_agg = pointnet_out.view(batch_size, 5, 1024).mean(dim=1)  # Mean pooling over the 5 dimension [batch_size, 1024]
                transformer_output_agg = transformer_out.view(16, 150, batch_size, 16).mean(dim=1)  # Mean pooling over the 150 dimension [16, batch_size, 16]
                transformer_output_agg_flat = transformer_output_agg.transpose(0, 1).reshape(batch_size, -1)  # [batch_size, 16*16]
                combined_features = torch.cat((pointnet_out_agg, transformer_output_agg_flat), dim=1)
                
                xw_1_pred, xw_2_pred, obj_catch_t_pred = mlp_model(combined_features)

                loss = criterion(xw_1_pred, xw_1_true) + criterion(xw_2_pred, xw_2_true) + criterion(obj_catch_t_pred, obj_catch_t_true)
                total_loss += loss.item()

                xw_1_pred = xw_1_pred.cpu().detach().numpy()  # Convert to numpy array if they are tensors
                xw_2_pred = xw_2_pred.cpu().detach().numpy()
                xw_1_pred = X_WG1_scaler.inverse_transform(xw_1_pred)
                xw_2_pred = X_WG2_scaler.inverse_transform(xw_2_pred)
                xw_1_true = xw_1_true.cpu()
                xw_2_true = xw_2_true.cpu()
                obj_catch_t_true = obj_catch_t_true.cpu().detach().numpy()
                obj_catch_t_pred = obj_catch_t_pred.cpu().detach().numpy()

                writer.writerow([
                        batch_idx, 
                        obj_catch_t_true, 
                        obj_catch_t_pred, 
                        xw_1_true.squeeze().tolist(), 
                        xw_1_pred.squeeze().tolist(), 
                        xw_2_true.squeeze().tolist(), 
                        xw_2_pred.squeeze().tolist()                   
                    ])

                print(f'idx:{batch_idx}, obj_catch_t_true:{obj_catch_t_true}, X_WG1_true:{xw_1_true}, X_WG2_true:{xw_2_true}')
                print(f'idx:{batch_idx}, obj_catch_t_pred:{obj_catch_t_pred}, X_WG1_predict:{xw_1_pred}, X_WG2_predict:{xw_2_pred}')
                print(f'loss:{torch.sqrt(loss)}')
            
            average_loss = total_loss / len(test_loader)
            print(f'Average loss on test set: {average_loss}')



if __name__ == '__main__':
    dataset = GraspDataset('graspnet_data_addtraj.h5')
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    pointnet_model = PointNetPlusPlus()
    # transformer_model = TrajTransformer(feature_size = 16, nhead = 8, num_encoder_layers = 4, num_decoder_layers = 4, dim_feedforward = 2048, max_seq_length = 16)
    transformer_model = TrajTransformer(feature_size = 16, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16)
    mlp_model = PredictionMLP(input_size = (1024+16*16), hidden_sizes = [512, 256, 128])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    pointnet_model = pointnet_model.to(device)
    transformer_model = transformer_model.to(device)
    mlp_model = mlp_model.to(device)

    optimizer = optim.Adam(
    list(pointnet_model.parameters()) + 
    list(transformer_model.parameters()) + 
    list(mlp_model.parameters()), 
    lr=0.001, 
    betas=(0.9, 0.999)
)
    criterion = nn.MSELoss()
    number_of_epoch = 100
    train_losses = []
    val_losses = []
    for epoch in range(number_of_epoch):
        train_loss, train_losses = train(pointnet_model, transformer_model, mlp_model, optimizer, criterion, train_loader, batch_size)
        val_loss, val_losses = val(pointnet_model, transformer_model, mlp_model, val_loader)

        print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}')

    torch.save(pointnet_model.state_dict(), 'C:/Users/51495/Desktop/thesis_research/pointnet_model_weights.pth')
    torch.save(transformer_model.state_dict(), 'C:/Users/51495/Desktop/thesis_research/transformer_model_weights.pth')
    torch.save(mlp_model.state_dict(), 'C:/Users/51495/Desktop/thesis_research/mlp_model_weights.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    #uncomment below for test
    # pointnet_model = PointNetPlusPlus()
    # transformer_model = TrajTransformer(feature_size = 16, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16)
    # mlp_model = PredictionMLP(input_size = (1024+16*16), hidden_sizes = [512, 256, 128])

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # pointnet_model = pointnet_model.to(device)
    # transformer_model = transformer_model.to(device)
    # mlp_model = mlp_model.to(device)
    # test(pointnet_model, transformer_model, mlp_model)
    