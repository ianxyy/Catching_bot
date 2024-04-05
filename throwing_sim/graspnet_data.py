from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    AbstractValue,
    PointCloud,
    RigidTransform,
    HydroelasticContactRepresentation,
    ContactResults,
    Rgba

)

import numpy as np
import h5py
from utils import ObjectTrajectory
from joblib import load
import torch
from torch_geometric.data import Data, Batch

from graspnet import PointNetPlusPlus, TrajTransformer, PredictionMLP


class GraspPredictor(LeafSystem):
    def __init__(self, original_plant, scene_graph, model_name, grasp_random_seed, velocity, roll, launching_position, launching_orientation, meshcat):
        LeafSystem.__init__(self)

        obj_traj = AbstractValue.Make(ObjectTrajectory())
        self.DeclareAbstractInputPort("object_trajectory", obj_traj)    #0 input
        obj_pc = AbstractValue.Make(PointCloud())
        self.DeclareAbstractInputPort("object_pc", obj_pc)              #1 input
        grasp = AbstractValue.Make({'gripper1': (RigidTransform(), 0), 'gripper2': (RigidTransform(), 0)}) # right:gripper1, left:gripper2
        self.DeclareAbstractInputPort("grasp_selection", grasp)         #2 input
        self.DeclareAbstractInputPort("contact_results_input", 
            AbstractValue.Make(ContactResults())                        #3 input
        )

        self.traj_count = 0
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.025, 0.0, self.data_collect)

        self.plant = original_plant
        self.graph = scene_graph
        self.model_name = model_name
        self.grasp_random_seed = grasp_random_seed

        self.traj_input = []
        self.pointcloud_input = []
        self.time_input = []
        self.X_WG1 = None
        self.X_WG2 = None
        self.obj_catch_t = None
        self.write = True

        self.traj_input_after = []
        self.time_input_after = []

        self.velocity = velocity
        self.roll = roll
        self.launching_position = launching_position
        self.launching_orientation = launching_orientation

        self.meshcat = meshcat

    def traj_collect(self, context):
        # traj_at_t = []
        # obj_traj = self.get_input_port(0).Eval(context)
        # for t in np.linspace(0, 1, 150):
        #     traj = obj_traj.value(t)
        #     traj_rotation = traj.rotation().matrix()
        #     traj_rotation = np.concatenate((traj_rotation[:,0].reshape(3,1),traj_rotation[:,1].reshape(3,1)), axis = None)  #6d rotation representation
        #     traj_translation = traj.translation()   #3x1
        #     traj_vel = obj_traj.EvalDerivative(t)   #6x1
        #     row = np.concatenate((traj_translation, traj_rotation, traj_vel), axis = None)  #15x1
        #     traj_at_t.append(row)    #(150,15,1)
        # traj_at_t = np.array(traj_at_t)
        # return traj_at_t
        obj_traj = self.get_input_port(0).Eval(context)
        traj_at_t = np.empty((150, 15))  # Preallocate array
        for i, t in enumerate(np.linspace(0, 1, 150)):
            traj = obj_traj.value(t)
            traj_at_t[i, :3] = traj.translation()
            traj_at_t[i, 3:9] = traj.rotation().matrix()[:, :2].reshape(-1)  # Flatten the first two columns
            traj_at_t[i, 9:] = obj_traj.EvalDerivative(t)
        return traj_at_t


    def pc_collect(self, context):
        target_num_points = 1024
        pc = self.get_input_port(1).Eval(context).VoxelizedDownSample(voxel_size=0.0025)
        pc = pc.xyzs().T      #Nx3

        num_points = pc.shape[0]
        if num_points == target_num_points:
            return pc
        elif num_points > target_num_points:
            # Randomly select 'target_num_points' from the point cloud
            indices = np.random.choice(num_points, target_num_points, replace=False)
            pc = pc[indices, :]
        else:
            # If fewer points than needed, duplicate some points
            indices = np.random.choice(num_points, target_num_points, replace=True)
            pc = np.vstack((pc, pc[indices, :]))  # Only append missing points
        # print('pc',np.shape(sampled_pc))        
        return pc


    # def check_success(self, context):
    #     contact_results = self.get_input_port(3).Eval(context)
        
    #     # No contacts mean no grasp
    #     if contact_results.num_point_pair_contacts() == 0:
    #         print('Grasp Failed: did not touch')
    #         return False
        
    #     # Define names of gripper parts and the robot body
    #     gripper_parts = ['left_finger', 'right_finger', 'body']
    #     obj_names = ['noodle', 'ring']  # Add more names if necessary
        
    #     # Iterate through all contact pairs
    #     for i in range(contact_results.num_point_pair_contacts()):
    #         contact_info = contact_results.point_pair_contact_info(i)
            
    #         # Get the names of the bodies involved in contact
    #         bodyA_name = self.plant.GetBodyFromFrameId(self.plant.GetBodyFrameIdOrThrow(contact_info.bodyA_index())).name()
    #         bodyB_name = self.plant.GetBodyFromFrameId(self.plant.GetBodyFrameIdOrThrow(contact_info.bodyB_index())).name()
            
    #         # Check if contact involves only the gripper parts and not the robot body
    #         if bodyA_name not in gripper_parts + obj_names or bodyB_name not in gripper_parts + obj_names:
    #             print('Grasp Failed: collide with robot body')
    #             return False
            
    #         # Check if the contact is between the gripper parts themselves
    #         if bodyA_name in gripper_parts and bodyB_name in gripper_parts:
    #             print('Grasp Failed: nothing between gripper fingers')
    #             return False
        
    #     # If none of the checks failed, the grasp is successful
    #     print('Grasp Succeed')
    #     return True
    def check_success(self, context):
        contact_results = self.get_input_port(3).Eval(context)
        
        # No contacts mean no grasp
        if contact_results.num_point_pair_contacts() == 0:
            print('Grasp Failed: did not touch')
            return False
        
        # Define names of gripper parts and the robot body
        gripper_1 = ['left_finger_1', 'right_finger_1', 'body_1']
        gripper_2 = ['left_finger_2', 'right_finger_2', 'body_2']
        obj_names = ['noodle']  # Add more names if necessary
        for i in range(300):
            obj_names.append(f'segment_{i}')
        bodyAB = []
        
        # Iterate through all contact pairs
        for i in range(contact_results.num_point_pair_contacts()):
            contact_info = contact_results.point_pair_contact_info(i)
            
            # Get the names of the bodies involved in contact
            bodyA_name = self.plant.GetBodyFromFrameId(self.plant.GetBodyFrameIdOrThrow(contact_info.bodyA_index())).name()
            bodyB_name = self.plant.GetBodyFromFrameId(self.plant.GetBodyFrameIdOrThrow(contact_info.bodyB_index())).name()
            AB = [bodyA_name, bodyB_name]
            
            # Check if contact involves only the gripper parts and not the robot body
            if bodyA_name not in gripper_1 + gripper_2 + obj_names or bodyB_name not in gripper_1 + gripper_2 + obj_names:
                print('Grasp Failed: collide with robot body')
                return False
            
            # Check if the contact is between the gripper parts themselves
            if ('left_finger_1' in AB or 'right_finger_1' in AB or 'left_finger_2' in AB or 'right_finger_2' in AB) and not (bodyA_name in obj_names or bodyB_name in obj_names):
                print('Grasp Failed: nothing between gripper fingers/fingers collide with other parts')
                return False
            
            bodyAB.append(bodyA_name)
            bodyAB.append(bodyB_name)

        gripper_1_contact_with_any_object = any(finger in bodyAB for finger in ['left_finger_1', 'right_finger_1']) and any(obj in bodyAB for obj in obj_names)
        gripper_2_contact_with_any_object = any(finger in bodyAB for finger in ['left_finger_2', 'right_finger_2']) and any(obj in bodyAB for obj in obj_names)
        if gripper_1_contact_with_any_object and gripper_2_contact_with_any_object:
            print("Grasp Succeed.")
            return True
        else:
            print("Grasp Failed: Both grippers are not effectively gripping the object.")
            return False


    def data_collect(self, context, state):
        grasp = self.get_input_port(2).Eval(context)
        _, self.obj_catch_t = grasp['gripper1']
        if context.get_time() >= self.obj_catch_t - 0.1 and context.get_time() < self.obj_catch_t - 0.05:
            self.X_WG1, _ = grasp['gripper1']
            self.X_WG2, _ = grasp['gripper2']
            # print(f'X_WG1:{self.X_WG1}, X_WG2:{self.X_WG2}')

        if self.traj_count < 5:
            traj = self.traj_collect(context)
            pointcloud = self.pc_collect(context)
            time = context.get_time()

            ## visualize pc input
            # cloud = PointCloud(pointcloud.T.shape[1])
            # if pointcloud.T.shape[1] > 0:
            #     cloud.mutable_xyzs()[:] = pointcloud.T
            # if self.meshcat is not None:
            #     print(f'pc drew')
            #     self.meshcat.SetObject(f"{str(self)}PointCloud{self.traj_count}", cloud, point_size=0.01, rgba=Rgba(0, 1.0, 0.5))

            self.traj_input.append(traj)    #5x150x15
            self.pointcloud_input.append(pointcloud)    #5x1024x3
            self.time_input.append(time)    #5x1    

            self.traj_count += 1
        elif self.traj_count >= 5 and self.traj_count <= 20:
            # self.model_pred()
            traj_after = self.traj_collect(context)
            time_after = context.get_time()

            self.traj_input_after.append(traj_after)    #16x150x15
            self.time_input_after.append(time_after)    #16x1 

            self.traj_count += 1
        else:
            if context.get_time() >= 0.8 and self.write:
                self.write = False 
                result = 1 if self.check_success(context) else 0
                #TODO write traj,pointcloud,time as input/ X_WG1,X_WG2,obj_catch_t as output in h5py
                X_WG1_rot = self.X_WG1.rotation().matrix()
                X_WG1_rotation = np.concatenate((X_WG1_rot[:,0].reshape(3,1),X_WG1_rot[:,1].reshape(3,1)), axis = None)  #6d rotation representation
                X_WG1_translation = self.X_WG1.translation()   #3x1
                X_WG1_input = np.concatenate((X_WG1_translation, X_WG1_rotation), axis = None)  #9x1

                X_WG2_rot = self.X_WG2.rotation().matrix()
                X_WG2_rotation = np.concatenate((X_WG2_rot[:,0].reshape(3,1),X_WG2_rot[:,1].reshape(3,1)), axis = None)  #6d rotation representation
                X_WG2_translation = self.X_WG2.translation()   #3x1
                X_WG2_input = np.concatenate((X_WG2_translation, X_WG2_rotation), axis = None)  #9x1
                self.write_data_to_h5py(X_WG1_input, X_WG2_input, result)
                self.clear_data_buffers()

    
    def write_data_to_h5py(self, X_WG1_input, X_WG2_input, result):
        if result == 1:
        # Create or open an HDF5 file
            with h5py.File('graspnet_data_addtraj.h5', 'a') as hf:
                # Check if datasets exist, and get the new index
                index = len(hf.keys()) // 9
                #input
                hf.create_dataset(f'traj_data_{index}', data=np.array(self.traj_input))     #5x150x15
                hf.create_dataset(f'pc_data_{index}', data=np.array(self.pointcloud_input)) #5x1024x3
                hf.create_dataset(f'time_data_{index}', data=np.array(self.time_input))     #5x1 
                hf.create_dataset(f'traj_data_after_{index}', data=np.array(self.traj_input_after))     #16x150x15
                hf.create_dataset(f'time_data_after_{index}', data=np.array(self.time_input_after))     #16x1 
                #output
                hf.create_dataset(f'X_WG1_{index}', data=np.array(X_WG1_input))             #9x1
                hf.create_dataset(f'X_WG2_{index}', data=np.array(X_WG2_input))             #9x1
                hf.create_dataset(f'obj_catch_t_{index}', data=np.array(self.obj_catch_t))        #1
                hf.create_dataset(f'result_{index}', data=np.array(result))                  #1

        with open('results_addtraj.txt', "a") as text_file:
            # Write the information to the file
            text_file.write(f"Object: {self.model_name}, Seed: {self.grasp_random_seed}, Result: {result}, vel: {self.velocity}, pos:{self.launching_position}, ori:{self.launching_orientation}\n")

    
    def clear_data_buffers(self):
        # Reset data buffers for the next batch of data collection
        self.traj_input = []
        self.pointcloud_input = []
        self.time_input = []
        self.traj_input_after = []
        self.time_input_after = []


    def model_pred(self):
        if self.traj_count == 5:
            with torch.no_grad():
                batch_size = 1
                pointnet_model = PointNetPlusPlus()
                transformer_model = TrajTransformer(feature_size = 16, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16)
                mlp_model = PredictionMLP(input_size = (1024+16*16), hidden_sizes = [512, 256, 128])

                # Load the trained weights
                pointnet_model.load_state_dict(torch.load('model/pointnet_model_weights.pth', map_location=torch.device('cpu')))
                transformer_model.load_state_dict(torch.load('model/transformer_model_weights.pth', map_location=torch.device('cpu')))
                mlp_model.load_state_dict(torch.load('model/mlp_model_weights.pth', map_location=torch.device('cpu')))
                
                # Switch all models to evaluation mode
                pointnet_model.eval()
                transformer_model.eval()
                mlp_model.eval()

                traj_scaler_path = 'model/traj_scaler.joblib'
                pc_scaler_path = 'model/pc_scaler.joblib'
                X_WG1_scaler_path = 'model/X_WG1_scaler.joblib'
                X_WG2_scaler_path = 'model/X_WG2_scaler.joblib'

                traj_scaler = load(traj_scaler_path)
                pc_scaler = load(pc_scaler_path)
                X_WG1_scaler = load(X_WG1_scaler_path)
                X_WG2_scaler = load(X_WG2_scaler_path)

                traj_data = np.array(self.traj_input)
                pc_data = np.array(self.pointcloud_input)
                time_data = np.array(self.time_input)

                traj_input_normalized = traj_scaler.transform(traj_data.reshape(-1, traj_data.shape[-1])).reshape(traj_data.shape)
                time_data_expanded = time_data[:, None, None]
                time_data_replicated = np.repeat(time_data_expanded, traj_input_normalized.shape[1], axis=1)
                traj_data_with_time = np.concatenate([traj_input_normalized, time_data_replicated], axis=2)

                pointcloud_input_normalized = pc_scaler.transform(pc_data.reshape(-1, pc_data.shape[-1])).reshape(pc_data.shape)
                
                traj_input_normalized_tensor = torch.tensor(traj_data_with_time, dtype=torch.float32).to('cpu')
                pointcloud_input_normalized_tensor = torch.tensor(pointcloud_input_normalized, dtype=torch.float32).to('cpu')
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
                xw_1_pred = xw_1_pred.cpu().detach().numpy()  # Convert to numpy array if they are tensors
                xw_2_pred = xw_2_pred.cpu().detach().numpy()
                xw_1_pred = X_WG1_scaler.inverse_transform(xw_1_pred)
                xw_2_pred = X_WG2_scaler.inverse_transform(xw_2_pred)

                print(f'obj_catch_t_pred:{obj_catch_t_pred}, X_WG1_predict:{xw_1_pred}, X_WG2_predict:{xw_2_pred}')
                
                    

