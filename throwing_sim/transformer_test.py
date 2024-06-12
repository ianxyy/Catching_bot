import h5py
import numpy as np
import math

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
import torch_geometric.transforms as T
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
        self.traj_pos_scaler = MaxAbsScaler()
        self.traj_vel_scaler = MaxAbsScaler()
        self.pc_scaler = MaxAbsScaler()

        self.X_WG1_pos_scaler = MaxAbsScaler()
        self.X_WG2_pos_scaler = MaxAbsScaler()
        self.catch_pos_scaler = MaxAbsScaler()
        if mode == 'train':
            self.traj_pos_after_scaler = MaxAbsScaler()
            self.traj_vel_after_scaler = MaxAbsScaler()
            self.compute_normalization_parameters()          

    def compute_normalization_parameters(self):
        with h5py.File(self.h5_file, 'r') as hf:
            all_traj_pos_data = []
            all_traj_vel_data = []
            all_traj_pos_data_after = []
            all_traj_vel_data_after = []
            all_pc_data = []
            all_time_data = []
            all_X_WG1_data = []
            all_X_WG2_data = []
            all_catch_pos_data = []
            for i in range(len(hf.keys()) // 10):
                traj_pos_data = hf[f'traj_data_{i}'][4,:,0:3]
                traj_vel_data = hf[f'traj_data_{i}'][4,:,9:12]
                traj_pos_data_after = hf[f'traj_data_after_{i}'][:,:,0:3]
                traj_vel_data_after = hf[f'traj_data_after_{i}'][:,:,9:12]
                pc_data = hf[f'pc_data_{i}'][4,:,:]
                # time_data = hf[f'time_data_{i}'][:]
                X_WG1_data = hf[f'X_WG1_{i}'][0:3]
                X_WG2_data = hf[f'X_WG2_{i}'][0:3]
                catch_pos_data = hf[f'obj_pose_at_catch_{i}'][:]
                all_traj_pos_data.append(traj_pos_data)
                all_traj_vel_data.append(traj_vel_data)
                all_traj_pos_data_after.append(traj_pos_data_after)
                all_traj_vel_data_after.append(traj_vel_data_after)
                all_pc_data.append(pc_data)
                # print('traj_data',traj_data.shape)
                # all_time_data.append(time_data)
                all_X_WG1_data.append(X_WG1_data)
                all_X_WG2_data.append(X_WG2_data)
                all_catch_pos_data.append(catch_pos_data)
            all_traj_pos_data = np.concatenate(all_traj_pos_data, axis=0)
            all_traj_vel_data = np.concatenate(all_traj_vel_data, axis=0)
            # print('all_traj_data',np.array(all_catch_pos_data).shape)
            # print('all_traj_data',all_traj_data.reshape(-1, all_traj_data.shape[-1]).shape)
            all_traj_pos_data_after = np.concatenate(all_traj_pos_data_after, axis=0)
            all_traj_vel_data_after = np.concatenate(all_traj_vel_data_after, axis=0)
            all_pc_data = np.concatenate(all_pc_data, axis=0)
            
            # all_time_data = np.concatenate(all_time_data, axis=0).reshape(-1, all_time_data[0].shape[-1])
            # all_X_WG1_data = np.concatenate(all_X_WG1_data, axis=0).reshape(-1, all_X_WG1_data[0].shape[-1])
            # all_X_WG2_data = np.concatenate(all_X_WG2_data, axis=0).reshape(-1, all_X_WG2_data[0].shape[-1])
            self.traj_pos_scaler.fit(all_traj_pos_data.reshape(-1, all_traj_pos_data.shape[-1]))
            self.traj_vel_scaler.fit(all_traj_vel_data.reshape(-1, all_traj_vel_data.shape[-1]))
            self.traj_pos_after_scaler.fit(all_traj_pos_data_after.reshape(-1, all_traj_pos_data_after.shape[-1]))
            self.traj_vel_after_scaler.fit(all_traj_vel_data_after.reshape(-1, all_traj_vel_data_after.shape[-1]))
            self.pc_scaler.fit(all_pc_data.reshape(-1, all_pc_data.shape[-1]))
            # self.time_scaler.fit(all_time_data)
            self.X_WG1_pos_scaler.fit(all_X_WG1_data)
            self.X_WG2_pos_scaler.fit(all_X_WG2_data)
            self.catch_pos_scaler.fit(all_catch_pos_data)

    def len(self):
        with h5py.File(self.h5_file, 'r') as hf:
            num_types_per_sample = 8 if self.mode == 'test' else 10
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
                catch_pos_data = hf[f'obj_pose_at_catch_{idx}'][:]

                return {
                    'pc' : torch.tensor(pc_data, dtype=torch.float32),
                    'traj': torch.tensor(traj_data, dtype=torch.float32),             #5,150,16
                    'time': torch.tensor(time_data, dtype=torch.float32),
                    'X_WG1': torch.tensor(X_WG1_data, dtype=torch.float32),
                    'X_WG2': torch.tensor(X_WG2_data, dtype=torch.float32),
                    'obj_catch_t': torch.tensor(obj_catch_t_data, dtype=torch.float32).unsqueeze(0),
                    'result': torch.tensor(result_data, dtype=torch.float32).unsqueeze(0),
                    'catch_pos': torch.tensor(catch_pos_data, dtype=torch.float32),
                }
            
            else:
                # Load data
                traj_data = hf[f'traj_data_{idx}'][:]
                # traj_data_after = hf[f'traj_data_after_{idx}'][:]
                pc_data = hf[f'pc_data_{idx}'][4,:,:]
                time_data = hf[f'time_data_{idx}'][:]
                time_after_data = hf[f'time_data_after_{idx}'][:]
                X_WG1_data = hf[f'X_WG1_{idx}'][:]
                X_WG2_data = hf[f'X_WG2_{idx}'][:]
                obj_catch_t_data = hf[f'obj_catch_t_{idx}'][()]
                result_data = hf[f'result_{idx}'][()]
                catch_pos_data = hf[f'obj_pose_at_catch_{idx}'][:]
                
                traj_pos_data = traj_data[4,:,0:3]
                traj_vel_data = traj_data[4,:,9:12]
                # traj_pos_data_after = traj_data_after[:,:,0:3]
                # traj_vel_data_after = traj_data_after[:,:,9:12]
                X_WG1_pos_data = X_WG1_data[0:3]
                X_WG2_pos_data = X_WG2_data[0:3]

                # Normalize trajectory data
                traj_pos_data_normalized = self.traj_pos_scaler.transform(traj_pos_data.reshape(-1, traj_pos_data.shape[-1])).reshape(traj_pos_data.shape)
                traj_vel_data_normalized = self.traj_vel_scaler.transform(traj_vel_data.reshape(-1, traj_vel_data.shape[-1])).reshape(traj_vel_data.shape)
                # traj_pos_data_after_normalized = self.traj_pos_after_scaler.transform(traj_pos_data_after.reshape(-1, traj_pos_data_after.shape[-1])).reshape(traj_pos_data_after.shape)
                # traj_vel_data_after_normalized = self.traj_vel_after_scaler.transform(traj_vel_data_after.reshape(-1, traj_vel_data_after.shape[-1])).reshape(traj_vel_data_after.shape)

                traj_data_normalized = traj_data[4,:,0:12]
                traj_data_normalized[:,0:3] = traj_pos_data_normalized
                traj_data_normalized[:,9:12] = traj_vel_data_normalized
                # traj_data_after_normalized = traj_data_after[:,:,0:12]
                # traj_data_after_normalized[:,:,0:3] = traj_pos_data_after_normalized
                # traj_data_after_normalized[:,:,9:12] = traj_vel_data_after_normalized

                # Time embedding
                # time_embeddings = self.time_embedding(torch.tensor(time_data))
                # time_after_embeddings = self.time_embedding(torch.tensor(time_after_data))
                # time_embeddings_expanded = time_embeddings.unsqueeze(1).repeat(1, 150, 1)               #5x150x16
                # time_after_embeddings_expanded = time_after_embeddings.unsqueeze(1).repeat(1, 150, 1)   #16x150x16
                # print(time_after_embeddings_expanded.size())
                # print(time_after_embeddings_expanded)
                # print(time_embeddings.size())

                # Normalize point cloud data
                pc_data_normalized = self.pc_scaler.transform(pc_data.reshape(-1, pc_data.shape[-1])).reshape(pc_data.shape)

                # Standardize gripper data
                X_WG1_pos_data_normalized = self.X_WG1_pos_scaler.transform(X_WG1_pos_data.reshape(-1, X_WG1_pos_data.shape[-1])).reshape(X_WG1_pos_data.shape)
                X_WG2_pos_data_normalized = self.X_WG2_pos_scaler.transform(X_WG2_pos_data.reshape(-1, X_WG2_pos_data.shape[-1])).reshape(X_WG2_pos_data.shape)
                X_WG1_data_normalized = X_WG1_data
                X_WG1_data_normalized[0:3] = X_WG1_pos_data_normalized
                X_WG2_data_normalized = X_WG2_data
                X_WG2_data_normalized[0:3] = X_WG2_pos_data_normalized

                catch_pos_data_normalized = self.catch_pos_scaler.transform(catch_pos_data.reshape(-1, catch_pos_data.shape[-1])).reshape(catch_pos_data.shape)
                
                # Save scaler 
                dump(self.traj_pos_scaler, 'model/traj_pos_scaler.joblib')
                dump(self.traj_vel_scaler, 'model/traj_vel_scaler.joblib')
                # dump(self.traj_pos_after_scaler, 'model/traj_pos_after_scaler.joblib')
                # dump(self.traj_vel_after_scaler, 'model/traj_vel_after_scaler.joblib')
                dump(self.pc_scaler, 'model/pc_scaler.joblib')
                dump(self.X_WG1_pos_scaler, 'model/X_WG1_scaler.joblib')
                dump(self.X_WG2_pos_scaler, 'model/X_WG2_scaler.joblib')
                dump(self.catch_pos_scaler, 'model/catch_pos_scaler.joblib')

                # Convert to tensors
                # traj_tensor= torch.tensor(traj_data_normalized, dtype=torch.float32)
                # traj_after_tensor= torch.tensor(traj_data_after_normalized, dtype=torch.float32)
                traj_input_tensor = torch.tensor(traj_data_normalized, dtype=torch.float32)

                # traj_with_time_tensor= torch.cat((traj_tensor, time_embeddings_expanded), dim=2)
                # traj_after_with_time_after_tensor = torch.cat((traj_after_tensor, time_after_embeddings_expanded), dim=2)
                pc_tensor = torch.tensor(pc_data_normalized, dtype=torch.float32)
                X_WG1_tensor = torch.tensor(X_WG1_data_normalized, dtype=torch.float32)
                X_WG2_tensor = torch.tensor(X_WG2_data_normalized, dtype=torch.float32)
                obj_catch_t_tensor = torch.tensor(obj_catch_t_data, dtype=torch.float32).unsqueeze(0)
                result_tensor = torch.tensor(result_data, dtype=torch.float32).unsqueeze(0)
                catch_pos_tensor = torch.tensor(catch_pos_data_normalized, dtype=torch.float32)
                # print(f'traj_data_with_time_normalized shape:{traj_with_time_tensor[4,100,12:]}')
                print(f'pc_data_normalized shape:{pc_tensor.shape}')
                # print(f'time_data shape:{time_data_normalized.shape}')
                # print(f'X_WG1 shape:{X_WG1_tensor.shape}')
                # print(f'X_WG2 shape:{X_WG2_tensor.shape}')
                # print(f'obj_catch_t_data shape:{obj_catch_t_tensor.shape}')
                # print(f'result_data shape:{result_tensor.shape}')

                # pointnet_input = [Data(pos = pc_tensor[i]) for i in range(pc_tensor.size(0))]
                # pointnet_batch = Batch.from_data_list(pointnet_input)
                pointnet_input = Data(pos = pc_tensor)
                pointnet_batch = Batch.from_data_list([pointnet_input])
                # print('size',pointnet_batch.size())

                return {
                    'pointnet_input' : pointnet_batch,
                    'traj_input': traj_input_tensor,
                    # 'transformer_input_src': traj_tensor,#traj_with_time_tensor,              #5,150,16
                    # 'transformer_input_tgt': traj_after_tensor, #traj_after_with_time_after_tensor,  #16,150,16
                    'pc': pc_tensor,
                    'X_WG1': X_WG1_tensor,
                    'X_WG2': X_WG2_tensor,
                    'obj_catch_t': obj_catch_t_tensor,
                    'result': result_tensor,
                    'catch_pos': catch_pos_tensor,
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
        self.global_sa_module = GlobalSAModule(Sequential(Linear(128 + 3, 256), ReLU(), Linear(256, 256)))

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        # sa2_out = self.sa2_module(*sa1_out)
        # x, pos, batch = self.global_sa_module(*sa2_out)
        x, pos, batch = self.sa2_module(*sa1_out)
        print('ori', sa0_out[1].size())
        print('sa1', sa1_out[0].size())
        return x, pos, batch


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
            # if hs == hidden_sizes[0]:
            #     layers.append(nn.BatchNorm1d(512))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.2))
            input_size = hs  # Next layer's input size is the current layer's output size
        
        # Final layer without ReLU to allow for negative values and more flexibility
        self.features = nn.Sequential(*layers)
        self.output_catch_pos = nn.Linear(hidden_sizes[-1], 3)  # Assuming gripper pose is a 3D vector

    def forward(self, x):
        features = self.features(x)
        catch_pos = self.output_catch_pos(features)
        return catch_pos

def train(pointnet_model, transformer_model, mlp_model, optimizer, criterion, trainloader, batch_size, log=True):
    pointnet_model.train()
    transformer_model.train()
    mlp_model.train()
    train_loss = 0.0
    transformer_feature_changes = []
    for batch_idx, batch in enumerate(trainloader):
        if batch['X_WG1'].size(0) < batch_size:
            break

        batch['pointnet_input'] = batch['pointnet_input'].to(device)   
        traj_input = batch['traj_input'].to(device)
        catch_pos = batch['catch_pos'].to(device)
        print('traj_batch', traj_input.size())

        src_transformer = traj_input.transpose(0, 1)
        transformer_out = transformer_model(src_transformer, tgt = None) 
        transformer_feature_changes.append(torch.var(transformer_out).item())

        transformer_output_agg = transformer_out.transpose(0, 1).mean(dim=1)

        pointnet_out, pos, batch = pointnet_model(batch['pointnet_input']) 
        print('size:', pointnet_out.size())

        point_out_scaler = MaxAbsScaler()
        transfomer_out_scaler = MaxAbsScaler()
        pointnet_out = torch.tensor(point_out_scaler.fit_transform(pointnet_out.cpu().detach().numpy()), dtype=torch.float32).to(device)
        transformer_out = torch.tensor(transfomer_out_scaler.fit_transform(transformer_output_agg.cpu().detach().numpy()), dtype=torch.float32).to(device)

        combined_features = torch.cat((pointnet_out, transformer_out), dim=1)

        optimizer.zero_grad()
        catch_pos_pred = mlp_model(combined_features)
        loss = criterion(catch_pos_pred,catch_pos)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss = train_loss/ (batch_idx + 1)
    train_losses.append(train_loss)
    average_transformer_change = sum(transformer_feature_changes) / len(transformer_feature_changes)
    transformer_variance.append(average_transformer_change)

    return train_loss, train_losses, average_transformer_change, transformer_variance

def val(pointnet_model, transformer_model, mlp_model, val_loader):
    pointnet_model.eval()
    transformer_model.eval()
    mlp_model.eval()  # Evaluation mode
    val_loss = 0.0
    transformer_feature_changes = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch['X_WG1'].size(0) < batch_size:
                break
            batch['pointnet_input'] = batch['pointnet_input'].to(device)
            traj_input = batch['traj_input'].to(device)
            catch_pos = batch['catch_pos'].to(device)

            src_transformer = traj_input.transpose(0, 1)
            transformer_out = transformer_model(src_transformer, tgt = None) 
            transformer_feature_changes.append(torch.var(transformer_out).item())
            transformer_output_agg = transformer_out.transpose(0, 1).mean(dim=1)

            pointnet_out = pointnet_model(batch['pointnet_input'])  

            point_out_scaler = MaxAbsScaler()
            transfomer_out_scaler = MaxAbsScaler()
            pointnet_out = torch.tensor(point_out_scaler.fit_transform(pointnet_out.cpu().detach().numpy()), dtype=torch.float32).to(device)
            transformer_out = torch.tensor(transfomer_out_scaler.fit_transform(transformer_output_agg.cpu().detach().numpy()), dtype=torch.float32).to(device)

            combined_features = torch.cat((pointnet_out, transformer_out), dim=1)

            catch_pos_pred = mlp_model(combined_features)
            loss = criterion(catch_pos_pred,catch_pos)
            val_loss += loss.item()
    val_loss = val_loss/ (batch_idx + 1)
    val_losses.append(val_loss)
    average_transformer_change = sum(transformer_feature_changes) / len(transformer_feature_changes)
    transformer_val_variance.append(average_transformer_change)
    return val_loss, val_losses, average_transformer_change, transformer_val_variance

def test(pointnet_model, transformer_model, mlp_model):
    batch_size = 1
    test_dataset = GraspDataset('transformer_test.h5', 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    pointnet_model.load_state_dict(torch.load('model/pointnet_model_weights.pth', map_location=torch.device('cuda')))
    transformer_model.load_state_dict(torch.load('model/transformer_model_weights.pth', map_location=torch.device('cuda')))
    mlp_model.load_state_dict(torch.load('model/mlp_model_weights.pth', map_location=torch.device('cuda')))
    pointnet_model.eval()
    transformer_model.eval()
    mlp_model.eval()  # Evaluation mode    

    traj_pos_scaler_path = 'model/traj_pos_scaler.joblib'
    traj_vel_scaler_path = 'model/traj_vel_scaler.joblib'
    pc_scaler_path = 'model/pc_scaler.joblib'
    catch_pos_path = 'model/catch_pos_scaler.joblib'

    traj_pos_scaler = load(traj_pos_scaler_path)
    traj_vel_scaler = load(traj_vel_scaler_path)
    pc_scaler = load(pc_scaler_path)
    catch_pos_scaler = load(catch_pos_path)

    criterion = torch.nn.MSELoss()  # Define the loss function
    total_loss = 0.0

    with torch.no_grad(): 
        for batch_idx, batch in enumerate(test_loader):
            import time
            start_time = time.time()
            traj_data = batch['traj'].numpy().squeeze()
            catch_pos_data = batch['catch_pos'].numpy()
            pc_data =batch['pc'].numpy().squeeze() 
            
            traj_pos_data = traj_data[4,:,0:3]
            traj_vel_data = traj_data[4,:,9:12]
            pc_data = pc_data[4,:,:]

            catch_pos_data_normalized = torch.tensor(catch_pos_scaler.transform(catch_pos_data.reshape(-1, catch_pos_data.shape[-1])).reshape(catch_pos_data.shape)).to(device)

            traj_pos_input_normalized = traj_pos_scaler.transform(traj_pos_data.reshape(-1, traj_pos_data.shape[-1])).reshape(traj_pos_data.shape)
            traj_vel_input_normalized = traj_vel_scaler.transform(traj_vel_data.reshape(-1, traj_vel_data.shape[-1])).reshape(traj_vel_data.shape)

            traj_data_normalized = traj_data[4,:,0:12]
            traj_data_normalized[:,0:3] = traj_pos_input_normalized
            traj_data_normalized[:,9:12] = traj_vel_input_normalized
            traj_tensor = torch.tensor(traj_data_normalized, dtype=torch.float32).to(device)

            pointcloud_input_normalized = pc_scaler.transform(pc_data.reshape(-1, pc_data.shape[-1])).reshape(pc_data.shape)
                
            pointcloud_input_normalized_tensor = torch.tensor(pointcloud_input_normalized, dtype=torch.float32).to('cuda')

            timesteps, features = traj_tensor.size()
            src_transformer = traj_tensor.view(batch_size, timesteps, features).transpose(0, 1).to(device)
            transformer_out = transformer_model(src_transformer, tgt = None) 
            transformer_output_agg_flat = transformer_out.transpose(0, 1).mean(dim=1)

            pointnet_input = Data(pos = pointcloud_input_normalized_tensor)
            pointnet_batch = Batch.from_data_list([pointnet_input])
            pointnet_out = pointnet_model(pointnet_batch)

            point_out_scaler = MaxAbsScaler()
            transfomer_out_scaler = MaxAbsScaler()
            pointnet_out = torch.tensor(point_out_scaler.fit_transform(pointnet_out.cpu()), dtype=torch.float32).to(device)
            transformer_out = torch.tensor(transfomer_out_scaler.fit_transform(transformer_output_agg_flat.cpu()), dtype=torch.float32).to(device)

            combined_features = torch.cat((pointnet_out, transformer_out), dim=1)
            catch_pos_pred = mlp_model(combined_features)
            
            loss = criterion(catch_pos_pred, catch_pos_data_normalized)
            total_loss += loss.item()

            catch_pos_pred = catch_pos_pred.cpu().detach().numpy()
            catch_pos_pred_inv_tran = catch_pos_scaler.inverse_transform(catch_pos_pred)



            print(f'idx:{batch_idx}, obj_catch_pos_true:{catch_pos_data}')
            print(f'idx:{batch_idx}, obj_catch_pos_pred:{catch_pos_pred_inv_tran}')
            print(f'loss:{torch.sqrt(loss)}')
            print(f'infer_time:{time.time() - start_time}')
        average_loss = total_loss / len(test_loader)
        print(f'Average loss on test set: {average_loss}')

def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

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
    dataset = GraspDataset('transformer_test.h5')
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    pointnet_model = PointNetPlusPlus()
    transformer_model = TrajTransformer(feature_size = 12, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16)
    # transformer_model = TrajTransformer(feature_size = 20, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16)
    mlp_model = PredictionMLP(input_size = (256+128), hidden_sizes = [512, 256, 128])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    transformer_variance = []
    transformer_val_variance = []
    early_stopping = EarlyStopping(patience=15, min_delta=0)
    for epoch in range(number_of_epoch):
        train_loss, train_losses, transformer_change, transformer_variances = train(pointnet_model, transformer_model, mlp_model, optimizer, criterion, train_loader, batch_size)
        val_loss, val_losses, t_val_change, t_val_variance = val(pointnet_model, transformer_model, mlp_model, val_loader)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Stopping early at epoch {epoch+1}")
            break

        print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}, Transformer Change = {transformer_change:.4f}, Transformer Val Change = {t_val_change:.4f}')

    torch.save(pointnet_model.state_dict(), '/home/haonan/Catching_bot/throwing_sim/model/pointnet_model_weights.pth')
    torch.save(transformer_model.state_dict(), '/home/haonan/Catching_bot/throwing_sim/model/transformer_model_weights.pth')
    torch.save(mlp_model.state_dict(), '/home/haonan/Catching_bot/throwing_sim/model/mlp_model_weights.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses_objframe_wo_norm')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_validation_losses_pointnet_w_norm.png')

    plt.figure(figsize=(10, 6))
    plt.plot(transformer_variances, label='training variance')
    plt.plot(t_val_variance, label='val variance')
    plt.title('Transformer Output Variance')
    plt.legend()
    plt.savefig('transformer_variance_pointnet_w_norm.png')

    
    #uncomment below for test
    # pointnet_model = PointNetPlusPlus()
    # transformer_model = TrajTransformer(feature_size = 12, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16)
    # mlp_model = PredictionMLP(input_size = (256+128), hidden_sizes = [512, 256, 128])

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # pointnet_model = pointnet_model.to(device)
    # transformer_model = transformer_model.to(device)
    # mlp_model = mlp_model.to(device)
    # test(pointnet_model, transformer_model, mlp_model)