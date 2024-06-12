import h5py
import numpy as np
import math
import csv

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, global_max_pool, fps, radius, knn_interpolate, MLP
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
from joblib import dump, load

class SinusoidalTimeEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, times):
        times = times.float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-math.log(10000.0) / self.embedding_dim))
        div_term = div_term.to(times.device)
        return torch.cat((torch.sin(times * div_term), torch.cos(times * div_term)), dim=-1)


class GraspDataset(Dataset):
    def __init__(self, h5_file, mode = 'train'):
        super(GraspDataset, self).__init__()
        self.h5_file = h5_file
        self.mode = mode
        self.traj_pos_scaler = MaxAbsScaler()
        self.traj_vel_scaler = MaxAbsScaler()
        self.pc_scaler = MaxAbsScaler()
        self.X_WG1_pos_scaler = MaxAbsScaler()
        self.X_WG2_pos_scaler = MaxAbsScaler()
        # self.X_OG1_pos_scaler = MaxAbsScaler()
        # self.X_OG2_pos_scaler = MaxAbsScaler()
        if mode == 'train':
            self.compute_normalization_parameters()          

    def compute_normalization_parameters(self):
        with h5py.File(self.h5_file, 'r') as hf:
            all_traj_pos_data = []
            all_traj_vel_data = []
            # all_traj_pos_data_after = []
            # all_traj_vel_data_after = []
            all_pc_data = []
            all_time_data = []
            all_X_WG1_data = []
            all_X_WG2_data = []
            # all_X_OG1_data = []
            # all_X_OG2_data = []
            for i in range(len(hf.keys()) // 14):
                # traj_pos_data = hf[f'traj_data_{i}'][:,:,0:3]
                # traj_vel_data = hf[f'traj_data_{i}'][:,:,9:12]
                traj_pos_data = hf[f'traj_data_{i}'][:,0:3]
                traj_vel_data = hf[f'traj_data_{i}'][:,9:12]
                # traj_pos_data_after = hf[f'traj_data_after_{i}'][:,:,0:3]
                # traj_vel_data_after = hf[f'traj_data_after_{i}'][:,:,9:12]
                pc_data = hf[f'pc_data_{i}'][:]
                # time_data = hf[f'time_data_{i}'][:]
                X_WG1_data = hf[f'X_WG1_tran_{i}']
                X_WG2_data = hf[f'X_WG2_tran_{i}']
                # X_OG1_data = hf[f'X_OG1_tran_{i}']
                # X_OG2_data = hf[f'X_OG2_tran_{i}']
                all_traj_pos_data.append(traj_pos_data)
                all_traj_vel_data.append(traj_vel_data)
                # all_traj_pos_data_after.append(traj_pos_data_after)
                # all_traj_vel_data_after.append(traj_vel_data_after)
                all_pc_data.append(pc_data)
                # print('traj_data',traj_data.shape)
                
                # all_time_data.append(time_data)
                all_X_WG1_data.append(X_WG1_data)
                all_X_WG2_data.append(X_WG2_data)
                # all_X_OG1_data.append(X_OG1_data)
                # all_X_OG2_data.append(X_OG2_data)
            all_traj_pos_data = np.concatenate(all_traj_pos_data, axis=0)
            all_traj_vel_data = np.concatenate(all_traj_vel_data, axis=0)
            # print('all_traj_data',all_traj_data.shape)
            # print('all_traj_data',all_traj_data.reshape(-1, all_traj_data.shape[-1]).shape)
            # all_traj_pos_data_after = np.concatenate(all_traj_pos_data_after, axis=0)
            # all_traj_vel_data_after = np.concatenate(all_traj_vel_data_after, axis=0)
            all_pc_data = np.concatenate(all_pc_data, axis=0)
            # all_time_data = np.concatenate(all_time_data, axis=0).reshape(-1, all_time_data[0].shape[-1])
            # all_X_WG1_data = np.concatenate(all_X_WG1_data, axis=0).reshape(-1, all_X_WG1_data[0].shape[-1])
            # all_X_WG2_data = np.concatenate(all_X_WG2_data, axis=0).reshape(-1, all_X_WG2_data[0].shape[-1])
            self.traj_pos_scaler.fit(all_traj_pos_data.reshape(-1, all_traj_pos_data.shape[-1]))
            self.traj_vel_scaler.fit(all_traj_vel_data.reshape(-1, all_traj_vel_data.shape[-1]))
            # self.traj_pos_after_scaler.fit(all_traj_pos_data_after.reshape(-1, all_traj_pos_data_after.shape[-1]))
            # self.traj_vel_after_scaler.fit(all_traj_vel_data_after.reshape(-1, all_traj_vel_data_after.shape[-1]))
            self.pc_scaler.fit(all_pc_data.reshape(-1, all_pc_data.shape[-1]))
            # self.time_scaler.fit(all_time_data)
            self.X_WG1_pos_scaler.fit(all_X_WG1_data)
            self.X_WG2_pos_scaler.fit(all_X_WG2_data)
            # self.X_OG1_pos_scaler.fit(all_X_OG1_data)
            # self.X_OG2_pos_scaler.fit(all_X_OG2_data)

            # Save scaler 
            dump(self.traj_pos_scaler, 'model/XW_ring_matrix_2/traj_pos_scaler.joblib')
            dump(self.traj_vel_scaler, 'model/XW_ring_matrix_2/traj_vel_scaler.joblib')
            # dump(self.traj_pos_after_scaler, 'model/traj_pos_after_scaler.joblib')
            # dump(self.traj_vel_after_scaler, 'model/traj_vel_after_scaler.joblib')
            dump(self.pc_scaler, 'model/XW_ring_matrix_2/pc_scaler.joblib')
            dump(self.X_WG1_pos_scaler, 'model/XW_ring_matrix_2/X_WG1_scaler.joblib')
            dump(self.X_WG2_pos_scaler, 'model/XW_ring_matrix_2/X_WG2_scaler.joblib')
            # dump(self.X_OG1_pos_scaler, 'model/XW_ring_matrix/X_OG1_scaler.joblib')
            # dump(self.X_OG2_pos_scaler, 'model/XW_ring_matrix/X_OG2_scaler.joblib')

    def len(self):
        with h5py.File(self.h5_file, 'r') as hf:
            num_types_per_sample = 14 if self.mode == 'test' else 14
            return len(hf.keys()) // num_types_per_sample

    def get(self, idx):
        with h5py.File(self.h5_file, 'r') as hf:
            if self.mode == 'test':
                # Load data
                traj_data = hf[f'traj_data_{idx}'][:]
                pc_data = hf[f'pc_data_{idx}'][:]
                # time_data = hf[f'time_data_{idx}'][:]
                X_WG1_data = hf[f'X_WG1_tran_{idx}'][:]
                X_WG2_data = hf[f'X_WG2_tran_{idx}'][:]
                X_WG1_rot_data = hf[f'X_WG1_rot_{idx}'][:]
                X_WG2_rot_data = hf[f'X_WG2_rot_{idx}'][:]
                # X_OG1_data = hf[f'X_OG1_tran_{idx}'][:]
                # X_OG2_data = hf[f'X_OG2_tran_{idx}'][:]
                # X_OG1_rot_data = hf[f'X_OG1_rot_{idx}'][:]
                # X_OG2_rot_data = hf[f'X_OG2_rot_{idx}'][:]
                obj_catch_t_data = hf[f'obj_catch_t_{idx}'][()]
                result_data = hf[f'result_{idx}'][()]

                return {
                    'pc' : torch.tensor(pc_data, dtype=torch.float32),
                    'traj': torch.tensor(traj_data, dtype=torch.float32),             #5,150,16
                    # 'time': torch.tensor(time_data, dtype=torch.float32),
                    'X_WG1_tran': torch.tensor(X_WG1_data, dtype=torch.float32),
                    'X_WG2_tran': torch.tensor(X_WG2_data, dtype=torch.float32),
                    'X_WG1_rot': torch.tensor(X_WG1_rot_data, dtype=torch.float32),
                    'X_WG2_rot': torch.tensor(X_WG2_rot_data, dtype=torch.float32),
                    # 'X_OG1_tran': torch.tensor(X_OG1_data, dtype=torch.float32),
                    # 'X_OG2_tran': torch.tensor(X_OG2_data, dtype=torch.float32),
                    # 'X_OG1_rot': torch.tensor(X_OG1_rot_data, dtype=torch.float32),
                    # 'X_OG2_rot': torch.tensor(X_OG2_rot_data, dtype=torch.float32),
                    'obj_catch_t': torch.tensor(obj_catch_t_data, dtype=torch.float32).unsqueeze(0),
                    'result': torch.tensor(result_data, dtype=torch.float32).unsqueeze(0),
                }
            
            else:
                # Load data
                traj_data = hf[f'traj_data_{idx}'][:]
                # traj_data_after = hf[f'traj_data_after_{idx}'][:]
                pc_data = hf[f'pc_data_{idx}'][:]
                # pc_normal = hf[f'pc_normal_{idx}'][:]
                # time_data = hf[f'time_data_{idx}'][:]
                # time_after_data = hf[f'time_data_after_{idx}'][:]
                X_WG1_data = hf[f'X_WG1_tran_{idx}'][:]
                X_WG2_data = hf[f'X_WG2_tran_{idx}'][:]
                X_WG1_rot_data = hf[f'X_WG1_rot_{idx}'][:]
                X_WG2_rot_data = hf[f'X_WG2_rot_{idx}'][:]
                # X_OG1_data = hf[f'X_OG1_tran_{idx}'][:]
                # X_OG2_data = hf[f'X_OG2_tran_{idx}'][:]
                # X_OG1_rot_data = hf[f'X_OG1_rot_{idx}'][:]
                # X_OG2_rot_data = hf[f'X_OG2_rot_{idx}'][:]
                obj_catch_t_data = hf[f'obj_catch_t_{idx}'][()]
                result_data = hf[f'result_{idx}'][()]
                
                # traj_pos_data = traj_data[:,:,0:3]
                # traj_vel_data = traj_data[:,:,9:12]
                traj_pos_data = traj_data[:,0:3]
                traj_vel_data = traj_data[:,9:12]

                # Normalize trajectory data
                traj_pos_data_normalized = self.traj_pos_scaler.transform(traj_pos_data.reshape(-1, traj_pos_data.shape[-1])).reshape(traj_pos_data.shape)
                traj_vel_data_normalized = self.traj_vel_scaler.transform(traj_vel_data.reshape(-1, traj_vel_data.shape[-1])).reshape(traj_vel_data.shape)

                # traj_data_normalized = traj_data[:,:,0:12]
                # traj_data_normalized[:,:,0:3] = traj_pos_data_normalized
                # traj_data_normalized[:,:,9:12] = traj_vel_data_normalized
                traj_data_normalized = traj_data[:,0:12]
                traj_data_normalized[:,0:3] = traj_pos_data_normalized
                traj_data_normalized[:,9:12] = traj_vel_data_normalized


                # Normalize point cloud data
                pc_data_normalized = self.pc_scaler.transform(pc_data.reshape(-1, pc_data.shape[-1])).reshape(pc_data.shape)

                # Standardize gripper data
                X_WG1_pos_data_normalized = self.X_WG1_pos_scaler.transform(X_WG1_data.reshape(-1, X_WG1_data.shape[-1])).reshape(X_WG1_data.shape)
                X_WG2_pos_data_normalized = self.X_WG2_pos_scaler.transform(X_WG2_data.reshape(-1, X_WG2_data.shape[-1])).reshape(X_WG2_data.shape)

                # X_OG1_pos_data_normalized = self.X_OG1_pos_scaler.transform(X_OG1_data.reshape(-1, X_OG1_data.shape[-1])).reshape(X_OG1_data.shape)
                # X_OG2_pos_data_normalized = self.X_OG2_pos_scaler.transform(X_OG2_data.reshape(-1, X_OG2_data.shape[-1])).reshape(X_OG2_data.shape)

                
                # Convert to tensors
                traj_tensor= torch.tensor(traj_data_normalized, dtype=torch.float32)
                # traj_after_tensor= torch.tensor(traj_data_after_normalized, dtype=torch.float32)

                # traj_with_time_tensor= torch.cat((traj_tensor, time_embeddings_expanded), dim=2)
                # traj_after_with_time_after_tensor = torch.cat((traj_after_tensor, time_after_embeddings_expanded), dim=2)
                pc_tensor = torch.tensor(pc_data_normalized, dtype=torch.float32)
                # pc_normal_tensor = torch.tensor(pc_normal, dtype=torch.float32)
                # Example usage
                # if check_for_nans(pc_normal_tensor):
                #     raise ValueError("NaN values found in input features.")
                X_WG1_tensor = torch.tensor(X_WG1_pos_data_normalized, dtype=torch.float32)
                X_WG2_tensor = torch.tensor(X_WG2_pos_data_normalized, dtype=torch.float32)
                # X_OG1_tensor = torch.tensor(X_OG1_pos_data_normalized, dtype=torch.float32)
                # X_OG2_tensor = torch.tensor(X_OG2_pos_data_normalized, dtype=torch.float32)

                X_WG1_rot_tensor = torch.tensor(X_WG1_rot_data, dtype=torch.float32)
                X_WG2_rot_tensor = torch.tensor(X_WG2_rot_data, dtype=torch.float32)
                # X_OG1_rot_tensor = torch.tensor(X_OG1_rot_data, dtype=torch.float32)
                # X_OG2_rot_tensor = torch.tensor(X_OG2_rot_data, dtype=torch.float32)
                obj_catch_t_tensor = torch.tensor(obj_catch_t_data, dtype=torch.float32).unsqueeze(0)
                result_tensor = torch.tensor(result_data, dtype=torch.float32).unsqueeze(0)
                # print(f'traj_data_with_time_normalized shape:{traj_with_time_tensor[4,100,12:]}')
                # print(f'pc_data_normalized shape:{pc_tensor.shape}')
                # print(f'time_data shape:{time_data_normalized.shape}')
                # print(f'X_WG1 shape:{X_WG1_tensor.shape}')
                # print(f'X_WG2 shape:{X_WG2_tensor.shape}')
                # print(f'obj_catch_t_data shape:{obj_catch_t_tensor.shape}')
                # print(f'result_data shape:{result_tensor.shape}')

                # pointnet_input = [Data(pos = pc_tensor[i]) for i in range(pc_tensor.size(0))]
                pointnet_input = [Data(pos = pc_tensor[i]) for i in range(pc_tensor.size(0))]
                pointnet_batch = Batch.from_data_list(pointnet_input)
                # print('size',pointnet_batch.size())

                return {
                    'pointnet_input' : pointnet_batch,
                    'transformer_input_src': traj_tensor,#traj_with_time_tensor,              #5,150,16
                    # 'transformer_input_tgt': traj_after_tensor, #traj_after_with_time_after_tensor,  #16,150,16
                    'pc': pc_tensor,
                    'X_WG1_tran': X_WG1_tensor,
                    'X_WG2_tran': X_WG2_tensor,
                    # 'X_OG1_tran': X_OG1_tensor,
                    # 'X_OG2_tran': X_OG2_tensor,
                    'X_WG1_rot': X_WG1_rot_tensor,
                    'X_WG2_rot': X_WG2_rot_tensor,
                    # 'X_OG1_rot': X_OG1_rot_tensor,
                    # 'X_OG2_rot': X_OG2_rot_tensor,
                    'obj_catch_t': obj_catch_t_tensor,
                    'result': result_tensor,
                }

def check_for_nans(tensor):
    if torch.isnan(tensor).any():
        print("NaN values found in tensor!")
        return True
    return False



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
        # self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        # self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        # self.global_sa_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        self.mlp = MLP([1024, 512, 256, 128], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        x, pos, batch = self.global_sa_module(*sa2_out)
        return self.mlp(x)


# class TrajTransformer(nn.Module):
#     def __init__(self, feature_size, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
#         super(TrajTransformer, self).__init__()
#         self.feature_size = feature_size
#         self.pos_encoder = PositionalEncoding(feature_size, max_seq_length)
#         self.transformer = nn.Transformer(d_model=feature_size, nhead=nhead,
#                                           num_encoder_layers=num_encoder_layers,
#                                           num_decoder_layers=num_decoder_layers,
#                                           dim_feedforward=dim_feedforward)
#         self.decoder = nn.Linear(feature_size, feature_size)
#         self.output_dropout = nn.Dropout(0.2)

#     def forward(self, src, tgt = None):
#         src = self.pos_encoder(src)
#         src = self.output_dropout(src)
#         if tgt is None:
#             #inference mode
#             batch_size = src.size(1)
#             pred_len = 16*150  # Number of steps to predict
#             tgt = torch.zeros(pred_len, batch_size, src.size(2), device=src.device)
            
#             # Iteratively predict future steps
#             for i in range(pred_len):
#                 tgt_temp = self.pos_encoder(tgt)
#                 output = self.transformer(src, tgt_temp)
#                 # Update tgt with the latest prediction
#                 tgt[i] = output[i]
#         else:
#             tgt = self.pos_encoder(tgt)
#             output = self.transformer(src, tgt)

#         output = self.output_dropout(output)
#         output = self.decoder(output)
#         return output

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.encoding = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         self.encoding[:, 0::2] = torch.sin(position * div_term)
#         self.encoding[:, 1::2] = torch.cos(position * div_term)
#         self.encoding = self.encoding.unsqueeze(0)

#     def forward(self, x):
#         return x + self.encoding[:, :x.size(1)].to(x.device)

class TrajTransformer(nn.Module):
    def __init__(self, feature_size, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
        super(TrajTransformer, self).__init__()
        self.feature_size = feature_size
        self.pos_encoder = PositionalEncoding(feature_size, max_seq_length)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead,
                                       dim_feedforward=dim_feedforward, dropout=0.1),
            num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(feature_size, 128)

    def forward(self, src, tgt = None):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # print(output)
        output = self.output_layer(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=150):
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
        self.output_gripper1_rot = self._build_mlp(hidden_sizes[-1], [256, 128, 64], 6)
        self.output_gripper2_rot = self._build_mlp(hidden_sizes[-1], [256, 128, 64], 6)
        self.output_gripper1_tran = self._build_mlp(hidden_sizes[-1], [256, 128, 64], 3)
        self.output_gripper2_tran = self._build_mlp(hidden_sizes[-1], [256, 128, 64], 3)
        self.output_catch_time = self._build_mlp(hidden_sizes[-1], [256, 128, 64], 1)

    def forward(self, x):
        features = self.features(x)
        xw_1_rot = self.output_gripper1_rot(features)
        xw_2_rot = self.output_gripper2_rot(features)
        xw_1_tran = self.output_gripper1_tran(features)
        xw_2_tran = self.output_gripper2_tran(features)
        obj_catch_t = self.output_catch_time(features)
        return xw_1_rot, xw_2_rot, xw_1_tran, xw_2_tran, obj_catch_t
    
    def _build_mlp(self, input_size, hidden_sizes, output_size):
        layers = []
        for hs in hidden_sizes:
            layers.append(nn.Linear(input_size, hs))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.3))
            input_size = hs
        layers.append(nn.Linear(input_size, output_size))
        return nn.Sequential(*layers)

def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def normalize_vectors(v):
    # """ Normalize batched vectors to avoid division by zero. """
    # norm = torch.norm(v, dim=-1, keepdim=True)
    # if torch.any(norm < 1e-6):
    #     print("Warning: Small norm detected")
    # return v / torch.clamp(norm, min=1e-6)  # Avoid division by zero
    return F.normalize(v, dim=-1)

def batch_6d_to_matrix(six_d):
    """
    Convert a batch of 6D rotation representations to rotation matrices.
    Args:
        six_d: (B, 6) tensor where each row contains two concatenated 3D vectors.
    Returns:
        matrices: (B, 3, 3) tensor of rotation matrices.
    """
    assert six_d.size(1) == 6, "Input tensor must have size (B, 6)"
    u = six_d[:, :3]
    v = six_d[:, 3:]

    u = normalize_vectors(u)
    # v = normalize_vectors(v)

    # Orthogonalize using the Gram-Schmidt process
    v = v - torch.sum(u * v, dim=-1, keepdim=True) * u
    v = normalize_vectors(v)

    # Compute the third vector via the cross product
    w = torch.cross(u, v, dim=-1)
    w = normalize_vectors(w)

    matrices = torch.stack([u, v, w], dim=-1)
    check_rotation_matrices(matrices)
    # print('matrix', matrices)
    matrices_9d = torch.cat([matrices[:, :, 0], matrices[:, :, 1], matrices[:, :, 2]], dim=-1)
    # print('9d', matrices_9d)
    return matrices_9d

def check_rotation_matrices(matrices):
    batch_size = matrices.size(0)
    for i in range(batch_size):
        mat = matrices[i].cpu().detach().numpy()
        should_be_identity = np.dot(mat.T, mat)
        I = np.eye(3)
        if not np.allclose(should_be_identity, I, atol=1e-6):
            print(f"Matrix {i} is not orthonormal!")


def minimum_rotation_angle(R1, R2):
    """
    Calculate the minimum rotation angle between two rotation matrices.
    Args:
        R1, R2: (B, 3, 3) PyTorch tensors of batched rotation matrices.
    Returns:
        angles: (B,) tensor of rotation angles in radians.
    """
    assert R1.size(1) == 3 and R1.size(2) == 3, "R1 must be of shape (B, 3, 3)"
    assert R2.size(1) == 3 and R2.size(2) == 3, "R2 must be of shape (B, 3, 3)"

    R_rel = torch.matmul(R1.transpose(1, 2), R2)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    trace = trace.clamp(-1.0, 3.0)

    angles = torch.acos((trace - 1) / 2)
    return angles

def angle_loss(xw_1_pred, xw_1_true):
    xw_1_rotv_pred = xw_1_pred[:,3:9]
    xw_1_rot_pred = batch_6d_to_matrix(xw_1_rotv_pred)

    xw_1_rotv_true = xw_1_true[:,3:9]
    xw_1_rot_true = batch_6d_to_matrix(xw_1_rotv_true)

    angle_diff = minimum_rotation_angle(xw_1_rot_pred, xw_1_rot_true)
    return torch.mean(angle_diff) #(torch.square(angle_diff))
    
def train(pointnet_model, transformer_model, mlp_model, optimizer, criterion, trainloader, batch_size, log=True):
    pointnet_model.train()
    transformer_model.train()
    mlp_model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(trainloader):
        # print(batch['X_WG1'].size(0))
        if batch['X_WG1_tran'].size(0) < batch_size:
            break
        batch['pointnet_input'] = batch['pointnet_input'].to(device)
        traj_input = batch['transformer_input_src'].to(device)
        # batch['transformer_input_tgt'] = batch['transformer_input_tgt'].to(device)
        xw_1_tran_true = batch['X_WG1_tran'].to(device)
        xw_2_tran_true = batch['X_WG2_tran'].to(device)
        xw_1_rot_true = batch['X_WG1_rot'].to(device)
        xw_2_rot_true = batch['X_WG2_rot'].to(device)
        # xo_1_tran_true = batch['X_OG1_tran'].to(device)
        # xo_2_tran_true = batch['X_OG2_tran'].to(device)
        # xo_1_rot_true = batch['X_OG1_rot'].to(device)
        # xo_2_rot_true = batch['X_OG2_rot'].to(device)
        obj_catch_t_true = batch['obj_catch_t'].to(device)
        
        pointnet_out = pointnet_model(batch['pointnet_input'])                          #[batch_size,1024]
        # print('size',batch['pointnet_input'].size())
        # src_transformer = reshape_for_transformer(batch['transformer_input_src'])       #[5x150,batch_size,16]
        # tgt_transformer = reshape_for_transformer(batch['transformer_input_tgt'])       #[16x150,batch_size,16]
        # transformer_out = transformer_model(src_transformer, tgt_transformer)           #[16x150,batch_size,16]
        src_transformer = traj_input.transpose(0, 1)
        transformer_out = transformer_model(src_transformer, tgt = None) 

        transformer_output_agg_flat = transformer_out.transpose(0, 1).mean(dim=1)
        
        pointnet_out_agg = pointnet_out.view(batch_size, 5, 128).mean(dim=1)  # Mean pooling over the 5 dimension [batch_size, 1024]
        # print('pointnet_out:',pointnet_out_agg)
        # transformer_output_agg = transformer_out.view(16, 150, batch_size, 12).mean(dim=1)  # Mean pooling over the 150 dimension [16, batch_size, 16]
        # transformer_output_agg_flat = transformer_output_agg.transpose(0, 1).reshape(batch_size, -1)  # [batch_size, 16*16]
        combined_features = torch.cat((pointnet_out_agg, transformer_output_agg_flat), dim=1)  # [batch_size, 1024 + 16*16]

        optimizer.zero_grad()
        xw_1_rot, xw_2_rot, xw_1_tran, xw_2_tran, obj_catch_t = mlp_model(combined_features)
        xw_1_matrix = batch_6d_to_matrix(xw_1_rot)
        xw_2_matrix = batch_6d_to_matrix(xw_2_rot)
        # print("Predicted 6D rotations:", xw_1_pred[:, 3:9])
        # print("True 6D rotations:", xw_1_true[:, 3:9])
        # print(xw_1_rot_true.shape)
        # print(xw_1_matrix.shape)
        # loss = criterion(xw_1_tran, xw_1_tran_true) + criterion(xw_2_tran, xw_2_tran_true) + criterion(xw_1_matrix, xw_1_rot_true) + criterion(xw_2_matrix, xw_2_rot_true) + criterion(obj_catch_t, obj_catch_t_true)
        loss = criterion(xw_1_tran, xw_1_tran_true) + criterion(xw_2_tran, xw_2_tran_true) + criterion(xw_1_matrix, xw_1_rot_true) + criterion(xw_2_matrix, xw_2_rot_true) + criterion(obj_catch_t, obj_catch_t_true)
        # loss = criterion(xw_1_tran, xw_1_tran_true) + criterion(xw_2_tran, xw_2_tran_true) + criterion(xw_1_rot, xw_1_rot_true[:,:6]) + criterion(xw_2_rot, xw_2_rot_true[:,:6]) + criterion(obj_catch_t, obj_catch_t_true)

        # loss = 0.1*criterion(xw_1_pred[:,:3], xw_1_true[:,:3]) + 0.1*criterion(xw_2_pred[:,:3], xw_2_true[:,:3]) + 0.375*angle_loss(xw_1_pred, xw_1_true) \
        #         + 0.375*angle_loss(xw_2_pred, xw_2_true) + 0.05*criterion(obj_catch_t_pred, obj_catch_t_true)
        # loss = criterion(xw_1_pred[:,:3], xw_1_true[:,:3]) + criterion(xw_2_pred[:,:3], xw_2_true[:,:3]) + angle_loss(xw_1_pred, xw_1_true) \
        #         + angle_loss(xw_2_pred, xw_2_true) + criterion(obj_catch_t_pred, obj_catch_t_true)
        # loss = 0.1 * criterion(xw_1_pred[:,:3], xw_1_true[:,:3]) + 0.1 * criterion(xw_2_pred[:,:3], xw_2_true[:,:3]) + \
        #     0.375 * criterion(xw_1_pred[:,3:9], xw_1_true[:,3:9]) + 0.375 * criterion(xw_2_pred[:,3:9], xw_2_true[:,3:9]) + 0.05 * criterion(obj_catch_t_pred, obj_catch_t_true)
        # print(f'pos_loss:{criterion(xw_1_pred[:,:3], xw_1_true[:,:3])}' )
        # print(f'rot_loss:{angle_loss(xw_1_pred, xw_1_true)}' )
        # loss = loss.mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss = train_loss/ (batch_idx + 1)
    train_losses.append(train_loss)
    return train_loss, train_losses

def val(pointnet_model, transformer_model, mlp_model, val_loader):
    pointnet_model.eval()
    transformer_model.eval()
    mlp_model.eval()  # Evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch['X_WG1_tran'].size(0) < batch_size:
                break
            batch['pointnet_input'] = batch['pointnet_input'].to(device)
            traj_input = batch['transformer_input_src'].to(device)
            # batch['transformer_input_tgt'] = batch['transformer_input_tgt'].to(device)
            xw_1_tran_true = batch['X_WG1_tran'].to(device)
            xw_2_tran_true = batch['X_WG2_tran'].to(device)
            xw_1_rot_true = batch['X_WG1_rot'].to(device)
            xw_2_rot_true = batch['X_WG2_rot'].to(device)
            # xo_1_tran_true = batch['X_OG1_tran'].to(device)
            # xo_2_tran_true = batch['X_OG2_tran'].to(device)
            # xo_1_rot_true = batch['X_OG1_rot'].to(device)
            # xo_2_rot_true = batch['X_OG2_rot'].to(device)
            obj_catch_t_true = batch['obj_catch_t'].to(device)
            
            pointnet_out = pointnet_model(batch['pointnet_input'])                          #[batch_size,1024]
            # src_transformer = reshape_for_transformer(batch['transformer_input_src'])       #[5x150,batch_size,16]
            # tgt_transformer = reshape_for_transformer(batch['transformer_input_tgt'])       #[16x150,batch_size,16]
            # transformer_out = transformer_model(src_transformer, tgt_transformer)           #[16x150,batch_size,16]
            src_transformer = traj_input.transpose(0, 1)
            transformer_out = transformer_model(src_transformer, tgt = None) 
            transformer_output_agg_flat = transformer_out.transpose(0, 1).mean(dim=1)

            pointnet_out_agg = pointnet_out.view(batch_size, 5, 128).mean(dim=1)  # Mean pooling over the 5 dimension [batch_size, 1024]
            # transformer_output_agg = transformer_out.view(16, 150, batch_size, 12).mean(dim=1)  # Mean pooling over the 150 dimension [16, batch_size, 16]
            # transformer_output_agg_flat = transformer_output_agg.transpose(0, 1).reshape(batch_size, -1)  # [batch_size, 16*16]
            combined_features = torch.cat((pointnet_out_agg, transformer_output_agg_flat), dim=1)  # [batch_size, 1024 + 16*16]

            xw_1_rot, xw_2_rot, xw_1_tran, xw_2_tran, obj_catch_t = mlp_model(combined_features)
            xw_1_matrix = batch_6d_to_matrix(xw_1_rot)
            xw_2_matrix = batch_6d_to_matrix(xw_2_rot)
            # print("Predicted 6D rotations:", xw_1_pred[:, 3:9])
            # print("True 6D rotations:", xw_1_true[:, 3:9])
            # loss = criterion(xw_1_tran, xw_1_tran_true) + criterion(xw_2_tran, xw_2_tran_true) + criterion(xw_1_matrix, xw_1_rot_true) + criterion(xw_2_matrix, xw_2_rot_true) + criterion(obj_catch_t, obj_catch_t_true)
            loss = criterion(xw_1_tran, xw_1_tran_true) + criterion(xw_2_tran, xw_2_tran_true) + criterion(xw_1_matrix, xw_1_rot_true) + criterion(xw_2_matrix, xw_2_rot_true) + criterion(obj_catch_t, obj_catch_t_true)
            # loss = criterion(xw_1_tran, xw_1_tran_true) + criterion(xw_2_tran, xw_2_tran_true) + criterion(xw_1_rot, xw_1_rot_true[:,:6]) + criterion(xw_2_rot, xw_2_rot_true[:,:6]) + criterion(obj_catch_t, obj_catch_t_true)
                # loss = 0.1*criterion(xw_1_pred[:,:3], xw_1_true[:,:3]) + 0.1*criterion(xw_2_pred[:,:3], xw_2_true[:,:3]) + 0.375*angle_loss(xw_1_pred, xw_1_true) \
                #     + 0.375*angle_loss(xw_2_pred, xw_2_true) + 0.05*criterion(obj_catch_t_pred, obj_catch_t_true)
                # loss = criterion(xw_1_pred[:,:3], xw_1_true[:,:3]) + criterion(xw_2_pred[:,:3], xw_2_true[:,:3]) + angle_loss(xw_1_pred, xw_1_true) \
                #     + angle_loss(xw_2_pred, xw_2_true) + criterion(obj_catch_t_pred, obj_catch_t_true)
                # loss = criterion(xw_1_pred, xw_1_true) + criterion(xw_2_pred, xw_2_true) + criterion(obj_catch_t_pred, obj_catch_t_true)
                # loss = 0.1 * criterion(xw_1_pred[:,:3], xw_1_true[:,:3]) + 0.1 * criterion(xw_2_pred[:,:3], xw_2_true[:,:3]) + \
                # 0.375 * criterion(xw_1_pred[:,3:9], xw_1_true[:,3:9]) + 0.375 * criterion(xw_2_pred[:,3:9], xw_2_true[:,3:9]) + 0.05 * criterion(obj_catch_t_pred, obj_catch_t_true)
                # loss = loss.mean()
            val_loss += loss.item()
    val_loss = val_loss/ (batch_idx + 1)
    val_losses.append(val_loss)
    return val_loss, val_losses
    
def batch_6d_to_matrix_2(six_d):
    """
    Convert a batch of 6D rotation representations to rotation matrices.
    Args:
    six_d: (B, 6) tensor where each row contains two concatenated 3D vectors.
    Returns:
    matrices: (B, 3, 3) tensor of rotation matrices.
    """
    # Ensure the input is of the correct shape
    assert six_d.size(1) == 6, "Input tensor must have size (B, 6)"

    # Split the 6D representation into two vectors of shape (B, 3)
    u = six_d[:, :3]
    v = six_d[:, 3:]

    u = normalize_vectors(u)
    v = normalize_vectors(v)
    # Directly use the cross product to ensure orthogonality
    w = torch.cross(u, v, dim=-1)
    w = normalize_vectors(w)

    # Recompute v to ensure orthogonality
    v = torch.cross(w, u, dim=-1)
    

    matrices = torch.stack([u, v, w], dim=-1)

    return matrices

def test(pointnet_model, transformer_model, mlp_model):
    batch_size = 1
    test_dataset = GraspDataset('graspnet9_data_ring_clear.h5', 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    pointnet_model.load_state_dict(torch.load('model/XW_ring_matrix/pointnet_model_weights.pth', map_location=torch.device('cuda')))
    transformer_model.load_state_dict(torch.load('model/XW_ring_matrix/transformer_model_weights.pth', map_location=torch.device('cuda')))
    mlp_model.load_state_dict(torch.load('model/XW_ring_matrix/mlp_model_weights.pth', map_location=torch.device('cuda')))

    pointnet_model.eval()
    transformer_model.eval()
    mlp_model.eval()  # Evaluation mode    

    traj_pos_scaler_path = 'model/XW_ring_matrix/traj_pos_scaler.joblib'
    traj_vel_scaler_path = 'model/XW_ring_matrix/traj_vel_scaler.joblib'
    pc_scaler_path = 'model/XW_ring_matrix/pc_scaler.joblib'
    X_WG1_scaler_path = 'model/XW_ring_matrix/X_WG1_scaler.joblib'
    X_WG2_scaler_path = 'model/XW_ring_matrix/X_WG2_scaler.joblib'
    # X_OG1_scaler_path = 'model/XW_ring_matrix/X_OG1_scaler.joblib'
    # X_OG2_scaler_path = 'model/XW_ring_matrix/X_OG2_scaler.joblib'


    traj_pos_scaler = load(traj_pos_scaler_path)
    traj_vel_scaler = load(traj_vel_scaler_path)
    pc_scaler = load(pc_scaler_path)
    X_WG1_scaler = load(X_WG1_scaler_path)
    X_WG2_scaler = load(X_WG2_scaler_path)
    # X_OG1_scaler = load(X_OG1_scaler_path)
    # X_OG2_scaler = load(X_OG2_scaler_path)


    criterion = torch.nn.MSELoss()  # Define the loss function
    total_loss = 0.0
    total_pos_loss = 0.0
    total_rot_loss = 0.0
    total_t_loss = 0.0
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(test_loader):
            traj_data = batch['traj'].numpy().squeeze() 
            pc_data =batch['pc'].numpy().squeeze() 
            # time_data = batch['time'].numpy().squeeze() 
            xw_1_tran_true = batch['X_WG1_tran'].numpy().squeeze() #.to('cuda')
            xw_2_tran_true = batch['X_WG2_tran'].numpy().squeeze() #.to('cuda')
            xw_1_rot_true = batch['X_WG1_rot'].numpy().squeeze() #.to('cuda')
            xw_2_rot_true = batch['X_WG2_rot'].numpy().squeeze()
            # xo_1_tran_true = batch['X_OG1_tran'].numpy().squeeze() #.to('cuda')
            # xo_2_tran_true = batch['X_OG2_tran'].numpy().squeeze() #.to('cuda')
            # xo_1_rot_true = batch['X_OG1_rot'].numpy().squeeze() #.to('cuda')
            # xo_2_rot_true = batch['X_OG2_rot'].numpy().squeeze()
            obj_catch_t_true = batch['obj_catch_t'].to('cuda')
            result_data = batch['result'].to('cuda')

            # traj_pos_data = traj_data[:,:,0:3]
            # traj_vel_data = traj_data[:,:,9:12]
            traj_pos_data = traj_data[:,0:3]
            traj_vel_data = traj_data[:,9:12]
            X_WG1_pos_data = xw_1_tran_true
            X_WG2_pos_data = xw_2_tran_true
            # X_OG1_pos_data = xo_1_tran_true
            # X_OG2_pos_data = xo_2_tran_true

            xw_1_pos_true_tensor = torch.tensor(X_WG1_scaler.transform(X_WG1_pos_data.reshape(-1, X_WG1_pos_data.shape[-1])).reshape(X_WG1_pos_data.shape), dtype=torch.float32).to('cuda')
            xw_2_pos_true_tensor = torch.tensor(X_WG2_scaler.transform(X_WG2_pos_data.reshape(-1, X_WG2_pos_data.shape[-1])).reshape(X_WG2_pos_data.shape), dtype=torch.float32).to('cuda')
            # xo_1_pos_true_tensor = torch.tensor(X_OG1_scaler.transform(X_OG1_pos_data.reshape(-1, X_OG1_pos_data.shape[-1])).reshape(X_OG1_pos_data.shape), dtype=torch.float32).to('cuda')
            # xo_2_pos_true_tensor = torch.tensor(X_OG2_scaler.transform(X_OG2_pos_data.reshape(-1, X_OG2_pos_data.shape[-1])).reshape(X_OG2_pos_data.shape), dtype=torch.float32).to('cuda')

            xw_1_rot_true_tensor = torch.tensor(xw_1_rot_true, dtype=torch.float32).to('cuda')
            xw_2_rot_true_tensor = torch.tensor(xw_2_rot_true, dtype=torch.float32).to('cuda')
            # xo_1_rot_true_tensor = torch.tensor(xo_1_rot_true, dtype=torch.float32).to('cuda')
            # xo_2_rot_true_tensor = torch.tensor(xo_2_rot_true, dtype=torch.float32).to('cuda')




            # print('xw1',xw_1_true)

            traj_pos_input_normalized = traj_pos_scaler.transform(traj_pos_data.reshape(-1, traj_pos_data.shape[-1])).reshape(traj_pos_data.shape)
            traj_vel_input_normalized = traj_vel_scaler.transform(traj_vel_data.reshape(-1, traj_vel_data.shape[-1])).reshape(traj_vel_data.shape)

            traj_data_normalized = traj_data[:,0:12]
            traj_data_normalized[:,0:3] = traj_pos_input_normalized
            traj_data_normalized[:,9:12] = traj_vel_input_normalized
            traj_tensor = torch.tensor(traj_data_normalized, dtype=torch.float32)
            # traj_with_time_tensor= torch.cat((traj_tensor, time_embeddings_expanded), dim=2).to('cuda')

            pointcloud_input_normalized = pc_scaler.transform(pc_data.reshape(-1, pc_data.shape[-1])).reshape(pc_data.shape)
            
            pointcloud_input_normalized_tensor = torch.tensor(pointcloud_input_normalized, dtype=torch.float32).to('cuda')
            # pointcloud_input_normalized_tensor = pointcloud_input_normalized_tensor.unsqueeze(0)  # Add batch dimension
            
            # print('shape', pc_data.shape)
            pointnet_input = [Data(pos = pointcloud_input_normalized_tensor[i]) for i in range(pointcloud_input_normalized_tensor.size(0))]
            pointnet_batch = Batch.from_data_list(pointnet_input)
            # print('batch_shape', pointnet_batch.size())

            src_transformer = traj_tensor.view(1,150,12).transpose(0, 1).to('cuda')

            pointnet_out = pointnet_model(pointnet_batch)
            transformer_out = transformer_model(src = src_transformer, tgt = None)

            pointnet_out_agg = pointnet_out.view(batch_size, 5, 128).mean(dim=1)  # Mean pooling over the 5 dimension [batch_size, 1024]
            # transformer_output_agg = transformer_out.view(16, 150, batch_size, 12).mean(dim=1)  # Mean pooling over the 150 dimension [16, batch_size, 16]
            # transformer_output_agg_flat = transformer_output_agg.transpose(0, 1).reshape(batch_size, -1)  # [batch_size, 16*16]
            transformer_output_agg_flat = transformer_out.transpose(0, 1).mean(dim=1)
            combined_features = torch.cat((pointnet_out_agg, transformer_output_agg_flat), dim=1)
            
            xw_1_rot, xw_2_rot, xw_1_tran, xw_2_tran, obj_catch_t = mlp_model(combined_features)
            xw_1_matrix = batch_6d_to_matrix(xw_1_rot).squeeze()
            xw_2_matrix = batch_6d_to_matrix(xw_2_rot).squeeze()
            # print(xw_2_matrix.size())
            # print(xw_1_rot_true_tensor.size())
            # loss = criterion(xw_1_pred.squeeze(), X_WG1_true_normalized) + criterion(xw_2_pred.squeeze(), X_WG2_true_normalized) + criterion(obj_catch_t_pred, obj_catch_t_true)
            # loss = 0.1*criterion(xw_1_pred[:,:3], X_WG1_true_normalized[:3]) + 0.1*criterion(xw_2_pred[:,:3], X_WG2_true_normalized[:3]) + 0.375*angle_loss(xw_1_pred, X_WG1_true_normalized.unsqueeze(0)) \
            # + 0.375*angle_loss(xw_2_pred, X_WG2_true_normalized.unsqueeze(0)) + 0.05*criterion(obj_catch_t_pred, obj_catch_t_true)

            # loss = criterion(xw_1_tran, xw_1_pos_true) + criterion(xw_2_tran, xw_2_pos_true) + criterion(xw_1_matrix, xw_1_rot_true) + criterion(xw_2_matrix, xw_2_rot_true) + criterion(obj_catch_t, obj_catch_t_true)
            loss = criterion(xw_1_tran, xw_1_pos_true_tensor) + criterion(xw_2_tran, xw_2_pos_true_tensor) + criterion(xw_1_matrix, xw_1_rot_true_tensor) + criterion(xw_2_matrix, xw_2_rot_true_tensor) + criterion(obj_catch_t, obj_catch_t_true)

            # print(f'pos_loss:{criterion(xw_1_pred[:,:3], X_WG1_true_normalized[:3])}' )
            # print(f'rot_loss:{angle_loss(xw_1_pred, X_WG1_true_normalized.unsqueeze(0))}' )
            xw_1_matrix = batch_6d_to_matrix_2(xw_1_rot)
            xw_2_matrix = batch_6d_to_matrix_2(xw_2_rot)
            xw_1_true = torch.stack([xw_1_rot_true_tensor[:3],xw_1_rot_true_tensor[3:6], xw_1_rot_true_tensor[6:9]], dim=-1).unsqueeze(0)
            xw_2_true = torch.stack([xw_2_rot_true_tensor[:3],xw_2_rot_true_tensor[3:6], xw_2_rot_true_tensor[6:9]], dim=-1).unsqueeze(0)
            total_loss += loss.item()
            total_pos_loss += (criterion(xw_1_tran, xw_1_pos_true_tensor) + criterion(xw_2_tran, xw_2_pos_true_tensor)) / 2 #(criterion(xw_1_pred[:,:3], X_WG1_true_normalized[:3]) + criterion(xw_2_pred[:,:3], X_WG2_true_normalized[:3])) / 2
            total_rot_loss += (minimum_rotation_angle(xw_1_matrix, xw_1_true) + minimum_rotation_angle(xw_2_matrix,xw_2_true)) /2 #(angle_loss(xw_1_pred, X_WG1_true_normalized.unsqueeze(0)) + angle_loss(xw_2_pred, X_WG2_true_normalized.unsqueeze(0))) / 2
            total_t_loss += criterion(obj_catch_t, obj_catch_t_true)

            xw_1_pred = xw_1_tran.cpu().detach().numpy()  # Convert to numpy array if they are tensors
            xw_2_pred = xw_2_tran.cpu().detach().numpy()
            # print(xw_1_pred) 
            xw_1_pred_tran = X_WG1_scaler.inverse_transform(xw_1_pred)
            xw_2_pred_tran = X_WG2_scaler.inverse_transform(xw_2_pred)

            obj_catch_t_true = obj_catch_t_true.cpu().detach().numpy()
            obj_catch_t_pred = obj_catch_t.cpu().detach().numpy()

            # print(f'idx:{batch_idx}, obj_catch_t_true:{obj_catch_t_true}, X_WG1_tran_true:{xw_1_pos_true}, X_WG2_tran_true:{xw_2_pos_true}, X_WG1_rot_true:{xw_1_rot_true}, X_WG2_rot_true:{xw_2_rot_true}')
            print(f'idx:{batch_idx}, obj_catch_t_true:{obj_catch_t_true}, X_WG1_tran_true:{xw_1_tran_true}, X_WG2_tran_true:{xw_2_tran_true}, X_WG1_rot_true:{xw_1_rot_true}, X_WG2_rot_true:{xw_2_rot_true}')
            print(f'idx:{batch_idx}, obj_catch_t_pred:{obj_catch_t_pred}, X_WG1_tran_predict:{xw_1_pred_tran}, X_WG2_tran_predict:{xw_2_pred_tran}, X_WG1_rot_predict:{xw_1_rot}, X_WG2_rot_predict:{xw_2_rot}')
            print(f'loss:{torch.sqrt(loss)}')
            print(f'rot_loss_1:{minimum_rotation_angle(xw_1_matrix, xw_1_true)} rot_loss_2:{minimum_rotation_angle(xw_2_matrix,xw_2_true)}')
            # print(f'matrix1:{xw_1_matrix}')
            print(f'matrix2:{xw_2_matrix}')
            print(f'matrix2:{xw_2_true}')
        
        average_loss = total_loss / len(test_loader)
        average_pos_loss = total_pos_loss / len(test_loader)
        average_rot_loss = total_rot_loss / len(test_loader)
        average_t_loss = total_t_loss / len(test_loader)
        print(f'Average loss on test set: {average_loss}')
        print(f'Average pos loss on test set: {average_pos_loss}')
        print(f'Average rotloss on test set: {average_rot_loss}')
        print(f'Average t loss on test set: {average_t_loss}')

class EarlyStopping:
    def __init__(self, patience, min_delta):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    dataset = GraspDataset('graspnet9_data_ring_clear.h5')
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    pointnet_model = PointNetPlusPlus()
    transformer_model = TrajTransformer(feature_size = 12, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16)
    # transformer_model = TrajTransformer(feature_size = 12, nhead = 4, num_encoder_layers = 4, num_decoder_layers = 4, dim_feedforward = 2048, max_seq_length = 16)
    # transformer_model = TrajTransformer(feature_size = 20, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16)
    mlp_model = PredictionMLP(input_size = (128+128), hidden_sizes = [512, 256, 128])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    pointnet_model = pointnet_model.to(device)
    transformer_model = transformer_model.to(device)
    mlp_model = mlp_model.to(device)

    optimizer = optim.Adam(
    list(pointnet_model.parameters()) + 
    list(transformer_model.parameters()) + 
    list(mlp_model.parameters()), 
    lr=0.0003, 
    betas=(0.9, 0.999)
)
    criterion = nn.MSELoss()
    number_of_epoch = 200
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=30, min_delta=0)
    for epoch in range(number_of_epoch):
        train_loss, train_losses = train(pointnet_model, transformer_model, mlp_model, optimizer, criterion, train_loader, batch_size)
        val_loss, val_losses = val(pointnet_model, transformer_model, mlp_model, val_loader)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Stopping early at epoch {epoch+1}")
            break

        print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}')

    torch.save(pointnet_model.state_dict(), '/home/haonan/Catching_bot/throwing_sim/model/XW_ring_matrix_2/pointnet_model_weights.pth')
    torch.save(transformer_model.state_dict(), '/home/haonan/Catching_bot/throwing_sim/model/XW_ring_matrix_2/transformer_model_weights.pth')
    torch.save(mlp_model.state_dict(), '/home/haonan/Catching_bot/throwing_sim/model/XW_ring_matrix_2/mlp_model_weights.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('ring_training_validation_losses.png')


    #uncomment below for test
    # pointnet_model = PointNetPlusPlus()
    # # transformer_model = TrajTransformer(feature_size = 20, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16)
    # transformer_model = TrajTransformer(feature_size = 12, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16)
    # mlp_model = PredictionMLP(input_size = (128+128), hidden_sizes = [512, 256, 128])
    # # mlp_model = PredictionMLP(input_size = (1024+16*20), hidden_sizes = [512, 256, 128])

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # pointnet_model = pointnet_model.to(device)
    # transformer_model = transformer_model.to(device)
    # mlp_model = mlp_model.to(device)
    # test(pointnet_model, transformer_model, mlp_model)
    