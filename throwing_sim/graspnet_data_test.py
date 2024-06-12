from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    AbstractValue,
    PointCloud,
    RigidTransform,
    HydroelasticContactRepresentation,
    ContactResults,
    Rgba,
    RotationMatrix,
    Sphere,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    RollPitchYaw,

)
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddMultibodyTriad
from manipulation.utils import ConfigureParser

import time
import numpy as np
import h5py
from utils import ObjectTrajectory
from joblib import load
import torch
from torch_geometric.data import Data, Batch
from scipy.spatial import cKDTree as KDTree
from torch_geometric.data import Data, Batch

# from graspnet_11_ring import PointNetPlusPlus, TrajTransformer, PredictionMLP, SinusoidalTimeEmbedding
from graspnet_11 import PointNetPlusPlus, TrajTransformer, PredictionMLP, SinusoidalTimeEmbedding


class GraspPredictor(LeafSystem):
    def __init__(self, original_plant, scene_graph, model_name, grasp_random_seed, velocity, roll, launching_position, launching_orientation, meshcat):
        LeafSystem.__init__(self)

        obj_traj = AbstractValue.Make(ObjectTrajectory())
        self.DeclareAbstractInputPort("object_trajectory", obj_traj)    #0 input
        obj_pc = AbstractValue.Make(PointCloud())
        self.DeclareAbstractInputPort("object_pc", obj_pc)              #1 input
        # grasp = AbstractValue.Make({'gripper1': (RigidTransform(), 0), 'gripper2': (RigidTransform(), 0)}) # right:gripper1, left:gripper2
        # self.DeclareAbstractInputPort("grasp_selection", grasp)         #2 input
        # self.DeclareAbstractInputPort("contact_results_input", 
            # AbstractValue.Make(ContactResults())                        #3 input
        # )

        port = self.DeclareAbstractOutputPort(
            "grasp_selection_output",
            lambda: AbstractValue.Make({'gripper1': (RigidTransform(), 0), 'gripper2': (RigidTransform(), 0)}),  # dict mapping grasp to a grasp time
            self.model_pred,
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

        self.collect_data  = False
        self.selected_grasp1_world_frame = None
        self.selected_grasp2_world_frame = None
        self.obj_catch_t_pred = None

    def draw_grasp_candidate(self, X_G1, X_G2, prefix="gripper", random_transform=True):
        """
        Helper function to visualize grasp.
        """
        # print('draw')
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        ConfigureParser(parser)
        # gripper_model_url = "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
        # gripper1_instance = parser.AddModelsFromUrl(gripper_model_url)[0]
        # gripper2_instance = parser.AddModelsFromUrl(gripper_model_url)[0]
        gripper_model_url = "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
        gripper_model_url2 = "package://manipulation/schunk_wsg_50_welded_fingers_copy.sdf"
        url = '/home/haonan/Catching_bot/throwing_sim/schunk_wsg_50_welded_fingers2.sdf'
        gripper1_instance = parser.AddModelsFromUrl(gripper_model_url)[0]
        # gripper2_instance = parser.AddModelsFromUrl(gripper_model_url2)[0]
        gripper2_instance = parser.AddModels(url)[0]

        if random_transform:
            X_G1 = self.random_transform @ X_G1
            X_G2 = self.random_transform @ X_G2

        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body", gripper1_instance), X_G1)
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body", gripper2_instance), X_G2)
        AddMultibodyTriad(plant.GetFrameByName("body", gripper1_instance), scene_graph)
        AddMultibodyTriad(plant.GetFrameByName("body", gripper2_instance), scene_graph)
        plant.Finalize()

        params = MeshcatVisualizerParams()
        params.prefix = prefix
        meshcat_vis = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, self.meshcat, params
        )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        diagram.ForcedPublish(context)

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
        traj_at_t = np.empty((150, 12))  # Preallocate array
        for i, t in enumerate(np.linspace(0, 1, 150)):
            traj = obj_traj.value(t)
            traj_at_t[i, :3] = traj.translation()
            traj_at_t[i, 3:9] = traj.rotation().matrix()[:, :2].reshape(-1)  # Flatten the first two columns
            traj_at_t[i, 9:] = obj_traj.EvalDerivative(t)[:3]
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
        # grasp = self.get_input_port(2).Eval(context)
        # _, self.obj_catch_t = grasp['gripper1']
        # if context.get_time() >= self.obj_catch_t - 0.1 and context.get_time() < self.obj_catch_t - 0.05:
        #     self.X_WG1, _ = grasp['gripper1']
        #     self.X_WG2, _ = grasp['gripper2']
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
        # else:
        #     if context.get_time() >= 0.8 and self.write:
        #         self.write = False 
        #         result = 1 if self.check_success(context) else 0
        #         #TODO write traj,pointcloud,time as input/ X_WG1,X_WG2,obj_catch_t as output in h5py
        #         # X_WG1_rot = self.X_WG1.rotation().matrix()
        #         # X_WG1_rotation = np.concatenate((X_WG1_rot[:,0].reshape(3,1),X_WG1_rot[:,1].reshape(3,1)), axis = None)  #6d rotation representation
        #         # X_WG1_translation = self.X_WG1.translation()   #3x1
        #         # X_WG1_input = np.concatenate((X_WG1_translation, X_WG1_rotation), axis = None)  #9x1

        #         # X_WG2_rot = self.X_WG2.rotation().matrix()
        #         # X_WG2_rotation = np.concatenate((X_WG2_rot[:,0].reshape(3,1),X_WG2_rot[:,1].reshape(3,1)), axis = None)  #6d rotation representation
        #         # X_WG2_translation = self.X_WG2.translation()   #3x1
        #         # X_WG2_input = np.concatenate((X_WG2_translation, X_WG2_rotation), axis = None)  #9x1
        #         self.write_data_to_h5py(X_WG1_input, X_WG2_input, result)
        #         self.clear_data_buffers()

    
    def write_data_to_h5py(self, X_WG1_input, X_WG2_input, result):
        if result == 1 and self.collect_data == True:
        # Create or open an HDF5 file
            with h5py.File('graspnet_data_addtraj_testing.h5', 'a') as hf:
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

        with open('results_addtraj_testing.txt', "a") as text_file:
            # Write the information to the file
            text_file.write(f"Object: {self.model_name}, Seed: {self.grasp_random_seed}, Result: {result}, vel: {self.velocity}, pos:{self.launching_position}, ori:{self.launching_orientation}\n")

    
    def clear_data_buffers(self):
        # Reset data buffers for the next batch of data collection
        self.traj_input = []
        self.pointcloud_input = []
        self.time_input = []
        self.traj_input_after = []
        self.time_input_after = []
        
    def normalize_vectors(self, v):
        """ Normalize batched vectors. """
        return v / torch.norm(v, dim=-1, keepdim=True)

    def batch_6d_to_matrix(self, six_d):
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

        # Normalize the vectors
        u = self.normalize_vectors(u)
        # v = self.normalize_vectors(v)

        # Ensure u and v are orthogonal by Gram-Schmidt process
        v = v - torch.sum(u * v, dim=-1, keepdim=True) * u
        v = self.normalize_vectors(v)

        # Compute the third vector as the cross product of u and v
        w = torch.cross(u, v, dim=-1)
        w = self.normalize_vectors(w)

        # Stack the vectors to form the rotation matrices
        matrices = torch.stack([u, v, w], dim=-1)
        self.check_rotation_matrices(matrices)

        # u = self.normalize_vectors(u)
        # v = self.normalize_vectors(v)
        # # Directly use the cross product to ensure orthogonality
        # w = torch.cross(u, v, dim=-1)
        # w = self.normalize_vectors(w)

        # # Recompute v to ensure orthogonality
        # v = torch.cross(w, u, dim=-1)
        

        # matrices = torch.stack([u, v, w], dim=-1)
        # self.check_rotation_matrices(matrices)
        return matrices
    
    def check_rotation_matrices(self, matrices):
        batch_size = matrices.size(0)
        for i in range(batch_size):
            mat = matrices[i].cpu().detach().numpy()
            should_be_identity = np.dot(mat.T, mat)
            I = np.eye(3)
            if not np.allclose(should_be_identity, I, atol=1e-6):
                print(f"Matrix {i} is not orthonormal!")
                raise ValueError("Matrix {i} is not orthonormal!")

    def model_pred(self, context, output):
        if self.traj_count < 5:
            print("collecting traj/pc data")
            return
        else:
            if self.selected_grasp1_world_frame == None:
                with torch.no_grad():
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    start_time = time.time()
                    batch_size = 1
                    pointnet_model = PointNetPlusPlus()
                    # transformer_model = TrajTransformer(feature_size = 12, nhead = 4, num_encoder_layers = 4, num_decoder_layers = 4, dim_feedforward = 2048, max_seq_length = 16)
                    transformer_model = TrajTransformer(feature_size = 12, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16)
                    # mlp_model = PredictionMLP(input_size = (1024+16*12), hidden_sizes = [512, 256, 128])
                    mlp_model = PredictionMLP(input_size = (128+128), hidden_sizes = [512, 256, 128])
                    # time_embedding = SinusoidalTimeEmbedding(embedding_dim = 8)

                    pointnet_model.to(device)
                    transformer_model.to(device)
                    mlp_model.to(device)

                    # Load the trained weights
                    pointnet_model.load_state_dict(torch.load('model/XW_matrix/pointnet_model_weights.pth', map_location=torch.device('cuda')))
                    transformer_model.load_state_dict(torch.load('model/XW_matrix/transformer_model_weights.pth', map_location=torch.device('cuda')))
                    mlp_model.load_state_dict(torch.load('model/XW_matrix/mlp_model_weights.pth', map_location=torch.device('cuda')))
                    # pointnet_model.load_state_dict(torch.load('model/XW_ring_matrix/pointnet_model_weights.pth', map_location=torch.device('cuda')))
                    # transformer_model.load_state_dict(torch.load('model/XW_ring_matrix/transformer_model_weights.pth', map_location=torch.device('cuda')))
                    # mlp_model.load_state_dict(torch.load('model/XW_ring_matrix/mlp_model_weights.pth', map_location=torch.device('cuda')))
                    
                    # Switch all models to evaluation mode
                    pointnet_model.eval()
                    transformer_model.eval()
                    mlp_model.eval()

                    traj_pos_scaler_path = 'model/XW_matrix/traj_pos_scaler.joblib'
                    traj_vel_scaler_path = 'model/XW_matrix/traj_vel_scaler.joblib'
                    pc_scaler_path = 'model/XW_matrix/pc_scaler.joblib'
                    X_WG1_scaler_path = 'model/XW_matrix/X_WG1_scaler.joblib'
                    X_WG2_scaler_path = 'model/XW_matrix/X_WG2_scaler.joblib'
                    # traj_pos_scaler_path = 'model/XW_ring_matrix/traj_pos_scaler.joblib'
                    # traj_vel_scaler_path = 'model/XW_ring_matrix/traj_vel_scaler.joblib'
                    # pc_scaler_path = 'model/XW_ring_matrix/pc_scaler.joblib'
                    # X_WG1_scaler_path = 'model/XW_ring_matrix/X_WG1_scaler.joblib'
                    # X_WG2_scaler_path = 'model/XW_ring_matrix/X_WG2_scaler.joblib'

                    traj_pos_scaler = load(traj_pos_scaler_path)
                    traj_vel_scaler = load(traj_vel_scaler_path)
                    pc_scaler = load(pc_scaler_path)
                    X_WG1_scaler = load(X_WG1_scaler_path)
                    X_WG2_scaler = load(X_WG2_scaler_path)

                    traj_data = np.array(self.traj_input)[4,:,:]
                    pc_data = np.array(self.pointcloud_input)
                    time_data = np.array(self.time_input)
                    traj_pos_data = traj_data[ :,0:3]
                    traj_vel_data = traj_data[ :,9:12]
                    traj_pos_input_normalized = traj_pos_scaler.transform(traj_pos_data.reshape(-1, traj_pos_data.shape[-1])).reshape(traj_pos_data.shape)
                    traj_vel_input_normalized = traj_vel_scaler.transform(traj_vel_data.reshape(-1, traj_vel_data.shape[-1])).reshape(traj_vel_data.shape)
                    # time_embeddings = time_embedding(torch.tensor(time_data).to(device))
                    # time_embeddings_expanded = time_embeddings.unsqueeze(1).repeat(1, 150, 1).to(device)               #5x150x16
                    traj_data_normalized = traj_data
                    traj_data_normalized[:,0:3] = traj_pos_input_normalized
                    traj_data_normalized[:,9:12] = traj_vel_input_normalized
                    # traj_tensor = torch.tensor(traj_data_normalized, dtype=torch.float32).to(device)
                    traj_with_time_tensor = torch.tensor(traj_data_normalized, dtype=torch.float32).to(device)
                    # traj_with_time_tensor= torch.cat((traj_tensor, time_embeddings_expanded), dim=2).to(device)
                    # print('traj_ori', traj_with_time_tensor)
                    pointcloud_input_normalized = pc_scaler.transform(pc_data.reshape(-1, pc_data.shape[-1])).reshape(pc_data.shape)
                    
                    # traj_input_normalized_tensor = torch.tensor(traj_data_with_time, dtype=torch.float32).to('cpu')
                    pointcloud_input_normalized_tensor = torch.tensor(pointcloud_input_normalized, dtype=torch.float32).to(device)
                    # pointcloud_input_normalized_tensor = pointcloud_input_normalized_tensor.unsqueeze(0)  # Add batch dimension
                    
                    # print('shape', pc_data.shape)
                    pointnet_input = [Data(pos = pointcloud_input_normalized_tensor[i]) for i in range(pointcloud_input_normalized_tensor.size(0))]
                    pointnet_batch = Batch.from_data_list(pointnet_input).to(device)
                    # pointnet_input = Data( pos = pointcloud_input_normalized_tensor)
                    # pointnet_batch = Batch.from_data_list([pointnet_input])
                    # print('batch_shape', pointnet_batch.size())

                    # timesteps, points, features = traj_with_time_tensor.size()
                    # # Reshape to [batch_size, timesteps * points, features]
                    # reshaped_data = traj_with_time_tensor.view(batch_size, timesteps * points, features)  
                    # # Transpose to match Transformer's expected input shape [seq_len, batch, features]
                    # src_transformer = reshaped_data.transpose(0, 1)
                    
                    
                    print(f'------------data-preprocessing time:{time.time() - start_time}')
                    pointnet_out = pointnet_model(pointnet_batch)
                    src_transformer = traj_with_time_tensor.view(1,150,12).transpose(0, 1).to('cuda')
                    transformer_out = transformer_model(src = src_transformer, tgt = None)
                    # print(f'transformer_size:{transformer_out.size()}')
                    transformer_output_agg_flat = transformer_out.view(150,1,128).transpose(0, 1).mean(dim=1)

                    pointnet_out_agg = pointnet_out.view(batch_size, 5, 128).mean(dim=1)  # Mean pooling over the 5 dimension [batch_size, 1024]
                    # transformer_output_agg = transformer_out.view(16, 150, batch_size, 12).mean(dim=1)  # Mean pooling over the 150 dimension [16, batch_size, 16]
                    # transformer_output_agg_flat = transformer_output_agg.transpose(0, 1).reshape(batch_size, -1)  # [batch_size, 16*16]
                    combined_features = torch.cat((pointnet_out_agg, transformer_output_agg_flat), dim=1)

                    # transformer_vis = transformer_out.view(150, 128).cpu().detach().numpy()
                    # print('traj_1', transformer_vis[75,:])
                    # print('traj_2', transformer_vis[120,:])
                    # print('traj_3', transformer_vis[50,:])
                    # print('traj_4', transformer_vis[140,:])
                    # if self.meshcat is not None:
                    #         for i in range(150):
                    #             self.meshcat.SetObject(f"RansacSpheres/{i}", Sphere(0.02), Rgba(1, 0, 1, 1))
                    #             self.meshcat.SetTransform(f"RansacSpheres/{i}", RigidTransform(RotationMatrix(), transformer_vis[i,0:3].reshape(3,1)))
                    #             # print(transformer_vis[t,i,0:3])
                    
                    # xw_1_pred, xw_2_pred, obj_catch_t_pred = mlp_model(combined_features)
                    xw_1_rot, xw_2_rot, xw_1_tran, xw_2_tran, obj_catch_t = mlp_model(combined_features)

                    rotation_matrix_xw_1 = self.batch_6d_to_matrix(xw_1_rot).squeeze(0).cpu().detach().numpy() 
                    rotation_matrix_xw_2 = self.batch_6d_to_matrix(xw_2_rot).squeeze(0).cpu().detach().numpy() 
                    print('rotation_matrix_xw_1', xw_1_rot)
                    print('rotation_matrix_xw_2', xw_2_rot)
                    xw_1_pred = xw_1_tran.cpu().detach().numpy()  # Convert to numpy array if they are tensors
                    xw_2_pred = xw_2_tran.cpu().detach().numpy()
                    # print(xw_1_pred) 
                    self.xw_1_pred_tran = X_WG1_scaler.inverse_transform(xw_1_pred)
                    self.xw_2_pred_tran = X_WG2_scaler.inverse_transform(xw_2_pred)
                    obj_catch_t_pred = obj_catch_t.cpu().detach().numpy()
                    # obj_catch_t_pred = obj_catch_t_pred.cpu().detach().numpy() 
                    self.obj_catch_t_pred = np.squeeze(obj_catch_t_pred)
                    # print(xw_1_pred[:,3:9].view(1,6).size())
                    # rotation_matrix_xw_1 = self.batch_6d_to_matrix(xw_1_pred[:,3:9].view(1,6)).squeeze(0).cpu().detach().numpy() 
                    # rotation_matrix_xw_2 = self.batch_6d_to_matrix(xw_2_pred[:,3:9].view(1,6)).squeeze(0).cpu().detach().numpy() 
                    # xw_1_pred = xw_1_pred.cpu().detach().numpy()  # Convert to numpy array if they are tensors
                    # xw_2_pred = xw_2_pred.cpu().detach().numpy()
                    # xw_1_pred_03 = X_WG1_scaler.inverse_transform(xw_1_pred[:,0:3])
                    # xw_2_pred_03 = X_WG2_scaler.inverse_transform(xw_2_pred[:,0:3])
                    # xw_1_pred[:,0:3]  = xw_1_pred_03
                    # xw_2_pred[:,0:3]  = xw_2_pred_03
                    
                    # v1 = xw_1_pred[:,3:6]
                    # v2 = xw_1_pred[:,6:9]
                    # v3 = np.cross(v1,v2)
                    # v3_normalized = v3 / np.linalg.norm(v3)
                    # rotation_matrix_xw_1 = np.column_stack((v1.reshape(3,1), v2.reshape(3,1), v3_normalized.reshape(3,1)))

                    # v1 = xw_2_pred[:,3:6]
                    # v2 = xw_2_pred[:,6:9]
                    # v3 = np.cross(v1,v2)
                    # v3_normalized = v3 / np.linalg.norm(v3)
                    # rotation_matrix_xw_2 = np.column_stack((v1.reshape(3,1), v2.reshape(3,1), v3_normalized.reshape(3,1)))
                    # residual_z = 0
                    obj_traj = self.get_input_port(0).Eval(context)
                    self.obj_pose_at_catch = obj_traj.value(self.obj_catch_t_pred)
                    print('pos:',self.obj_pose_at_catch.translation()[:3])
                    # if self.obj_pose_at_catch.translation()[2] > self.xw_1_pred_tran[:,2] and self.obj_pose_at_catch.translation()[2] > self.xw_2_pred_tran[:,2]:
                    #     max_z = np.max([self.xw_1_pred_tran[:,2], self.xw_2_pred_tran[:,2]])
                    #     residual_z = self.obj_pose_at_catch.translation()[2] - max_z
                    #     print('residual', residual_z)
                    residual_x = np.abs(self.xw_1_pred_tran[:,0] - self.xw_2_pred_tran[:,0])/2
                    if self.xw_1_pred_tran[:,0] > self.xw_2_pred_tran[:,0] :
                        residual_x1 = self.obj_pose_at_catch.translation()[0] + residual_x
                        residual_x2 = self.obj_pose_at_catch.translation()[0] - residual_x
                    else:
                        residual_x1 = self.obj_pose_at_catch.translation()[0] - residual_x
                        residual_x2 = self.obj_pose_at_catch.translation()[0] + residual_x
                        
                    print(self.xw_1_pred_tran, self.xw_2_pred_tran)
                    self.xw_1_pred_tran[:,0] = residual_x1
                    self.xw_2_pred_tran[:,0] = residual_x2
                    self.xw_1_pred_tran[:,2] = self.obj_pose_at_catch.translation()[2]
                    self.xw_2_pred_tran[:,2] = self.obj_pose_at_catch.translation()[2]
                    self.selected_grasp1_world_frame = RigidTransform(RotationMatrix(rotation_matrix_xw_1), self.xw_1_pred_tran.reshape(3,1))# + np.array([0, 0, residual_z]).reshape(3,1))
                    self.selected_grasp2_world_frame = RigidTransform(RotationMatrix(rotation_matrix_xw_2), self.xw_2_pred_tran.reshape(3,1))# + np.array([0, 0, residual_z]).reshape(3,1))
                    # self.draw_grasp_candidate(self.selected_grasp1_world_frame, self.selected_grasp2_world_frame, prefix="grippers_best1_obj", random_transform=False)
                    # rotation = RollPitchYaw(np.radians(-90), 0, 0).ToRotationMatrix()
                    # new_rotation = rotation @ self.selected_grasp1_world_frame.rotation()
                    # # Create a new RigidTransform with the new rotation and the old translation
                    # self.selected_grasp1_obj_frame = RigidTransform(new_rotation, rotation @ self.selected_grasp1_world_frame.translation())
                    # new_rotation_2 = rotation @ self.selected_grasp2_world_frame.rotation()
                    # # Create a new RigidTransform with the new rotation and the old translation
                    # self.selected_grasp2_obj_frame = RigidTransform(new_rotation_2, rotation @ self.selected_grasp2_world_frame.translation())
                    
                    # self.selected_grasp1_world_frame = self.obj_pose_at_catch @ self.selected_grasp1_obj_frame
                    # self.selected_grasp2_world_frame = self.obj_pose_at_catch @ self.selected_grasp2_obj_frame
                    # print('xw1', self.selected_grasp1_world_frame)
                    # print('xw2', self.selected_grasp2_world_frame)


                    print(f'------------infer time:{time.time() - start_time}')
                    obj_vel_at_catch = obj_traj.EvalDerivative(self.obj_catch_t_pred)[:3]
                    end_point = self.obj_pose_at_catch.translation()[:3] + 0.1 * obj_vel_at_catch
                    vertices = np.hstack([self.obj_pose_at_catch.translation().reshape(3, 1), end_point.reshape(3, 1)])
                    self.meshcat.SetLine("velocity_vector", vertices, 1.0, Rgba(r=1.0, g=0.0, b=0.0, a=1.0) )

                    print(f'obj_catch_t_pred:{self.obj_catch_t_pred}, X_WG1_predict:{self.selected_grasp1_world_frame}, X_WG2_predict:{self.selected_grasp2_world_frame}')
                    # self.draw_grasp_candidate(self.selected_grasp1_world_frame, self.selected_grasp2_world_frame, prefix="grippers_best1", random_transform=False)
                    output.set_value({'gripper1': (self.selected_grasp1_world_frame, self.obj_catch_t_pred), 'gripper2': (self.selected_grasp2_world_frame, self.obj_catch_t_pred)})
            else:
                
                obj_traj = self.get_input_port(0).Eval(context)
                # estimated_obj_catch_pose = RigidTransform(self.obj_pose_at_catch.rotation(), obj_traj.value(self.obj_catch_t_pred).translation())
                # self.selected_grasp1_world_frame = estimated_obj_catch_pose @ self.selected_grasp1_obj_frame
                # self.selected_grasp2_world_frame = estimated_obj_catch_pose @ self.selected_grasp2_obj_frame
                # self.draw_grasp_candidate(self.selected_grasp1_world_frame, self.selected_grasp2_world_frame, prefix="grippers_best1", random_transform=False)
                self.obj_pose_at_catch = obj_traj.value(self.obj_catch_t_pred)
                print('pos:',self.obj_pose_at_catch.translation()[:3])
                # residual_z = 0
                # if self.obj_pose_at_catch.translation()[2] > self.xw_1_pred_tran[:,2] and self.obj_pose_at_catch.translation()[2] > self.xw_2_pred_tran[:,2]:
                #     max_z = np.max([self.xw_1_pred_tran[:,2], self.xw_2_pred_tran[:,2]])
                #     residual_z = self.obj_pose_at_catch.translation()[2] - max_z
                #     print('residual', residual_z)
                residual_x = np.abs(self.xw_1_pred_tran[:,0] - self.xw_2_pred_tran[:,0])/2
                if self.xw_1_pred_tran[:,0] > self.xw_2_pred_tran[:,0] :
                    residual_x1 = self.obj_pose_at_catch.translation()[0] + residual_x
                    residual_x2 = self.obj_pose_at_catch.translation()[0] - residual_x
                else:
                    residual_x1 = self.obj_pose_at_catch.translation()[0] - residual_x
                    residual_x2 = self.obj_pose_at_catch.translation()[0] + residual_x
                    
                print(self.xw_1_pred_tran, self.xw_2_pred_tran)
                self.xw_1_pred_tran[:,0] = residual_x1
                self.xw_2_pred_tran[:,0] = residual_x2
                self.xw_1_pred_tran[:,2] = self.obj_pose_at_catch.translation()[2]
                self.xw_2_pred_tran[:,2] = self.obj_pose_at_catch.translation()[2]
                self.selected_grasp1_world_frame = RigidTransform(self.selected_grasp1_world_frame.rotation(), self.xw_1_pred_tran.reshape(3,1))# + np.array([0, 0, residual_z]).reshape(3,1))
                self.selected_grasp2_world_frame = RigidTransform(self.selected_grasp2_world_frame.rotation(), self.xw_2_pred_tran.reshape(3,1))# + np.array([0, 0, residual_z]).reshape(3,1))
                self.draw_grasp_candidate(self.selected_grasp1_world_frame, self.selected_grasp2_world_frame, prefix="grippers_best1", random_transform=False)
                output.set_value({'gripper1': (self.selected_grasp1_world_frame, self.obj_catch_t_pred), 'gripper2': (self.selected_grasp2_world_frame, self.obj_catch_t_pred)})



                    

