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

# from graspnet_2 import PointNetPlusPlus, TrajTransformer, PredictionMLP, SinusoidalTimeEmbedding, TrajMLP
from graspnet_5 import PointNetGlobal, PointNetPLocal, TrajTransformer, PredictionMLP, CombinedModel


class SpecificBodyPoseExtractor(LeafSystem):
    def __init__(self, plant, body_poses_output_port):
        LeafSystem.__init__(self)
        self.plant = plant
        self.body_index = plant.GetBodyByName('noodle').index()

        self.DeclareAbstractInputPort('body_poses', body_poses_output_port.Allocate())

        # self.DeclareVectorOutputPort('noodle_pose', (RigidTransform(), 0), self.CalcNoodleBodyPose)
        self.DeclareAbstractOutputPort(
            "noodle_pose",
            lambda: AbstractValue.Make((RigidTransform(), 0)),  # dict mapping grasp to a grasp time
            self.CalcNoodleBodyPose,
        )


    def CalcNoodleBodyPose(self, context, output):
        body_poses = self.EvalAbstractInput(context, 0).get_value()
        X_WB = body_poses[self.body_index]
        rpy = X_WB.rotation().ToRollPitchYaw()
        pose_vector = np.concatenate((X_WB.translation(), [rpy.roll_angle(), rpy.pitch_angle(), rpy.yaw_angle()]))
        print('rpy:',rpy)
        output.set_value(X_WB)

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

        self.traj = np.zeros(1)
        self.pointcloud = []
        self.pc_ori = []
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
        self.selected_grasp1_obj_frame = None
        self.selected_grasp2_obj_frame = None

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
        target_num_points = 1700
        pc_ori = self.get_input_port(1).Eval(context).VoxelizedDownSample(voxel_size=0.0025)
        pc = pc_ori.xyzs().T      #Nx3

        num_points = pc.shape[0]
        # print(f'num_points:{num_points}')
        if num_points == target_num_points:
            return pc
        elif num_points > target_num_points:
            # Randomly select 'target_num_points' from the point cloud
            indices = np.random.choice(num_points, target_num_points, replace=False)
            pc = pc[indices, :]
        else:
            # If fewer points than needed, duplicate some points
            indices = np.random.choice(num_points, target_num_points, replace=True)
            pc = pc[indices, :]
        # print('pc', pc.shape)  
        # kdtree = KDTree(pc)
        # _, index_1 = kdtree.query(first_grasp_point.reshape(1, -1), k=1)
        # _, index_2 = kdtree.query(second_grasp_point.reshape(1, -1), k=1)   
        # grasp1 = pc[index_1,:].T
        # grasp2 = pc[index_2,:].T

        return pc, pc_ori


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
        # print(f'time:{context.get_time()}, catch_t:{self.obj_catch_t}')

        if context.get_time() == 0.1: #self.traj_count < 5:
            self.traj = self.traj_collect(context)
            # print(self.traj)
            time = context.get_time()
            # self.X_WG1 = grasp['gripper1_pc']
            # self.X_WG2 = grasp['gripper2_pc']
            # print(f'X_WG1:{self.X_WG1}, X_WG2:{self.X_WG2}')
            # self.index_1 = grasp['index1']
            # self.index_2 = grasp['index2']
            # print(f'index_1:{self.index_1}, index_2:{self.index_2}')
            # X_WB = self.get_input_port(4).Eval(context)
            # print(f'X_WB{X_WB}')

            
            
            # self.point_1 = (X_WB @ self.X_WG1) #right gripper
            # self.point_2 = (X_WB @ self.X_WG2) #left
            self.pointcloud, self.pc_ori= self.pc_collect(context)
            # self.points = np.concatenate((self.point_1.reshape(1,3), self.point_2.reshape(1,3)) , axis = 0)
            # print(f'point_1:{self.point_1} point_2:{self.point_2} points:{self.points.shape}')
            # print(f'index_1:{self.index_1} index_2:{self.index_2} grasp1:{self.grasp1} grasp2:{self.grasp2}')
            # visualize pc input
            cloud = PointCloud(self.pointcloud.T.shape[1])
            if self.pointcloud.T.shape[1] > 0:
                cloud.mutable_xyzs()[:] = self.pointcloud.T
            if self.meshcat is not None:
            #     print(f'pc drew')
                self.meshcat.SetObject(f"{str(self)}PointCloud{self.traj_count}", cloud, point_size=0.01, rgba=Rgba(0, 1.0, 0.5))
            #     self.meshcat.SetObject(f"point/1", Sphere(0.005), Rgba(0.2, 0.2, 1, 1))
            #     self.meshcat.SetTransform(f"point/1", RigidTransform(self.point_1))
            #     self.meshcat.SetObject(f"point/2", Sphere(0.005), Rgba(0.2, 0.2, 1, 1))
            #     self.meshcat.SetTransform(f"point/2", RigidTransform(self.point_2))
            #     self.meshcat.SetObject(f"point/1pc", Sphere(0.005), Rgba(0.2, 0.2, 1, 1))
            #     self.meshcat.SetTransform(f"point/1pc", RigidTransform(self.X_WG1))
            #     self.meshcat.SetObject(f"point/2pc", Sphere(0.005), Rgba(0.2, 0.2, 1, 1))
            #     self.meshcat.SetTransform(f"point/2pc", RigidTransform(self.X_WG2))

            #     self.meshcat.SetObject(f"grasp1/1pc", Sphere(0.0025), Rgba(1.0, 0.2, 1, 1))
            #     self.meshcat.SetTransform(f"grasp1/1pc", RigidTransform(self.grasp1))
            #     self.meshcat.SetObject(f"grasp2/2pc", Sphere(0.0025), Rgba(1.0, 0.2, 1, 1))
            #     self.meshcat.SetTransform(f"grasp2/2pc", RigidTransform(self.grasp2))
            # self.grasp1 = self.grasp1.reshape(1,3)
            # self.grasp2 = self.grasp2.reshape(1,3)


        else:
            if context.get_time() >= 0.8 and self.write:
                self.write = False 
                result = 1 if self.check_success(context) else 0
                # self.write_data_to_h5py(self.traj, self.pointcloud, self.index_1, self.index_2, self.grasp1, self.grasp2, result)
                # self.clear_data_buffers()


    
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

    def model_pred(self, context, output):
        if self.traj.size == 1:
            print("collecting traj/pc data")
            return
        else:
            if self.selected_grasp1_world_frame == None:
                with torch.no_grad():
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    start_time = time.time()
                    batch_size = 1
                    pointnet_global = PointNetGlobal().to(device)
                    # poinet_local = PointNetPLocal().to(device)
                    transformer = TrajTransformer(feature_size = 12, nhead = 4, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, max_seq_length = 16).to(device)
                    trans_feature_mlp = PredictionMLP(input_size = (512+128), hidden_sizes = [512, 256, 128]).to(device)
                    combined_model = CombinedModel(pointnet_global, transformer,  trans_feature_mlp, feature_dim = 256+128 , transformer_output_dim = 128).to(device)

                    # Load the trained weights
                    pointnet_global.load_state_dict(torch.load('model/pointnet_global_weights.pth', map_location=torch.device('cuda')))
                    # poinet_local.load_state_dict(torch.load('model/poinet_local.pth', map_location=torch.device('cuda')))
                    transformer.load_state_dict(torch.load('model/transformer_weights.pth', map_location=torch.device('cuda')))
                    trans_feature_mlp.load_state_dict(torch.load('model/trans_feature_mlp_weights.pth', map_location=torch.device('cuda')))
                    combined_model.load_state_dict(torch.load('model/combined_model_weights.pth', map_location=torch.device('cuda')))

                    # Switch all models to evaluation mode
                    combined_model.eval()

                    traj_pos_scaler_path = 'model/traj_pos_scaler.joblib'
                    traj_vel_scaler_path = 'model/traj_vel_scaler.joblib'
                    pc_scaler_path = 'model/pc_scaler.joblib'
                    
                    traj_pos_scaler = load(traj_pos_scaler_path)
                    traj_vel_scaler = load(traj_vel_scaler_path)
                    pc_scaler = load(pc_scaler_path)

                    traj_data = np.array(self.traj)
                    pc_data = np.array(self.pointcloud)
                    # time_data = np.array(self.time_input)
                    traj_pos_data = traj_data[:,0:3]
                    traj_vel_data = traj_data[:,9:12]

                    # catch_pos_data_normalized = torch.tensor(target.transform(target.reshape(-1, target.shape[-1])).reshape(target.shape)).to(device)

                    traj_pos_input_normalized = traj_pos_scaler.transform(traj_pos_data.reshape(-1, traj_pos_data.shape[-1])).reshape(traj_pos_data.shape)
                    traj_vel_input_normalized = traj_vel_scaler.transform(traj_vel_data.reshape(-1, traj_vel_data.shape[-1])).reshape(traj_vel_data.shape)

                    traj_data_normalized = traj_data
                    traj_data_normalized[:,0:3] = traj_pos_input_normalized
                    traj_data_normalized[:,9:12] = traj_vel_input_normalized
                    traj_tensor = torch.tensor(traj_data_normalized, dtype=torch.float32).view(1,150,12).to(device)

                    pointcloud_input_normalized = pc_scaler.transform(pc_data.reshape(-1, pc_data.shape[-1])).reshape(pc_data.shape)
                        
                    pointcloud_input_normalized_tensor = torch.tensor(pointcloud_input_normalized, dtype=torch.float32).to('cuda')

                    pointnet_input = Data(pos = pointcloud_input_normalized_tensor)
                    pointnet_batch = Batch.from_data_list([pointnet_input])

                    print(f'------------data-preprocessing time:{time.time() - start_time}')
                    
                    point1, point2, obj_catch_t_pred = combined_model(pointnet_batch, traj_tensor)
                    
                    obj_catch_t_pred = obj_catch_t_pred.cpu().detach().numpy() 
                    self.obj_catch_t_pred = np.squeeze(obj_catch_t_pred)

                    point1 = point1.cpu().detach().numpy()  # Convert to numpy array if they are tensors
                    point2 = point2.cpu().detach().numpy()

                    point1_inv = pc_scaler.inverse_transform(point1[:,0:3])
                    point2_inv = pc_scaler.inverse_transform(point2[:,0:3])

                    print('point1', point1_inv)
                    print('point2', point2_inv)
                    

                    self.pc_ori.EstimateNormals(0.05, 30)
                    kdtree = KDTree(self.pc_ori.xyzs().T)
                    _, index_1 = kdtree.query(point1_inv.reshape(1, -1), k=1)
                    _, index_2 = kdtree.query(point2_inv.reshape(1, -1), k=1)
                    self.meshcat.SetObject(f"point/1", Sphere(0.005), Rgba(0.2, 0.2, 1, 1))
                    self.meshcat.SetTransform(f"point/1", RigidTransform(self.pointcloud[index_1,:].reshape(3,1)))
                    self.meshcat.SetObject(f"point/2", Sphere(0.005), Rgba(0.2, 0.2, 1, 1))
                    self.meshcat.SetTransform(f"point/2", RigidTransform(self.pointcloud[index_2,:].reshape(3,1)))
                    self.meshcat.SetObject(f"point/3", Sphere(0.005), Rgba(1.0, 0, 0, 1))
                    self.meshcat.SetTransform(f"point/3", RigidTransform(point1_inv.reshape(3,1)))
                    self.meshcat.SetObject(f"point/4", Sphere(0.005), Rgba(1.0, 0, 0, 1))
                    self.meshcat.SetTransform(f"point/4", RigidTransform(point2_inv.reshape(3,1)))
                    X_OF_1 = self.compute_darboux_frame(index_1, self.pc_ori, kdtree)
                    X_OF_2 = self.compute_darboux_frame(index_2, self.pc_ori, kdtree)
                    y_offset = -0.05
                    self.new_X_OG_1 = X_OF_1 @ RigidTransform(np.array([0, y_offset, 0]))  # Move gripper back by fixed amount
                    self.new_X_OG_2 = X_OF_2 @ RigidTransform(np.array([0, y_offset, 0]))

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

                    # self.selected_grasp1_obj_frame = RigidTransform(RotationMatrix(rotation_matrix_xw_1), xw_1_pred[:,0:3].reshape(3,1))
                    # self.selected_grasp2_obj_frame = RigidTransform(RotationMatrix(rotation_matrix_xw_2), xw_2_pred[:,0:3].reshape(3,1))
                    # self.draw_grasp_candidate(self.new_X_OG_1, self.new_X_OG_2, prefix="grippers_best", random_transform=False)
                    

                    obj_traj = self.get_input_port(0).Eval(context)
                    self.obj_pose_at_catch = obj_traj.value(self.obj_catch_t_pred)
                    self.selected_grasp1_world_frame = self.obj_pose_at_catch @ self.new_X_OG_1
                    self.selected_grasp2_world_frame = self.obj_pose_at_catch @ self.new_X_OG_2

                    print(f'------------infer time:{time.time() - start_time}')
                    print(f'obj_catch_t_pred:{self.obj_catch_t_pred}, X_WG1_predict:{self.selected_grasp1_world_frame}, X_WG2_predict:{self.selected_grasp2_world_frame}')
                    output.set_value({'gripper1': (self.selected_grasp1_world_frame, self.obj_catch_t_pred), 'gripper2': (self.selected_grasp2_world_frame, self.obj_catch_t_pred)})
            else:
                obj_traj = self.get_input_port(0).Eval(context)
                
                estimated_obj_catch_pose = RigidTransform(self.obj_pose_at_catch.rotation(), obj_traj.value(self.obj_catch_t_pred).translation())
                self.selected_grasp1_world_frame = estimated_obj_catch_pose @ self.new_X_OG_1
                self.selected_grasp2_world_frame = estimated_obj_catch_pose @ self.new_X_OG_2
                self.draw_grasp_candidate(self.selected_grasp1_world_frame, self.selected_grasp2_world_frame, prefix="grippers_best1", random_transform=False)
                output.set_value({'gripper1': (self.selected_grasp1_world_frame, self.obj_catch_t_pred), 'gripper2': (self.selected_grasp2_world_frame, self.obj_catch_t_pred)})

    def compute_darboux_frame(self, index, obj_pc, kdtree, ball_radius=0.002, max_nn=50):
            """
            Given a index of the pointcloud, return a RigidTransform from origin of
            point cloud to the Darboux frame at that point.

            Args:
            - index (int): index of the pointcloud.
            - obj_pc (PointCloud object): pointcloud of the object.
            - kdtree (scipy.spatial.KDTree object): kd tree to use for nn search.
            - ball_radius (float): ball_radius used for nearest-neighbors search
            - max_nn (int): maximum number of points considered in nearest-neighbors search.
            """
            points = obj_pc.xyzs()  # 3xN np array of points
            normals = obj_pc.normals()  # 3xN np array of normals

            # 1. Find nearest neighbors to point in PC
            nn_distances, nn_indices = kdtree.query(points[:,index].flatten(), max_nn, distance_upper_bound=ball_radius)
            finite_indices = np.isfinite(nn_distances)
            nn_indices = nn_indices[finite_indices]

            # 2. compute N, covariance matrix of all normal vectors in neighborhood
            nn_normals = normals[:, nn_indices]  # 3xK matrix where K is the number of neighbors the point has
            N = nn_normals @ nn_normals.T  # 3x3

            # 3. Eigen decomp (v1 = normal, v2 = major tangent, v3 = minor tangent)
            # The Eigenvectors create an orthogonal basis (note that N is symmetric) that can be used to construct a rotation matrix
            eig_vals, eig_vecs = np.linalg.eig(N)  # vertically stacked eig vecs
            # Sort the eigenvectors based on the eigenvalues
            sorted_indices = np.argsort(eig_vals)[::-1]  # Get the indices that would sort the eigenvalues in descending order
            eig_vals = eig_vals[sorted_indices]  # Sort the eigenvalues
            eig_vecs = eig_vecs[:, sorted_indices]  # Sort the eigenvectors accordingly

            # 4. Ensure v1 (eig vec corresponding to largest eig val) points into object
            if (eig_vecs[:,0] @ normals[:,index] > 0):  # if dot product with normal is pos, that means v1 is pointing out
                eig_vecs[:,0] *= -1  # flip v1

            # 5. Construct Rotation matrix to X_WF (by horizontal stacking v2 v1 v3)
            # This works bc rotation matrices are, by definition, 3 horizontally stacked orthonormal columns
            # Also, we choose the order [v2 v1 v3] bc v1 (with largest eigen value) corresponds to y-axis, v2 (with 2nd largest eigen value) corresponds to major axis of curvature (x-axis), and v3 (smallest eignvalue) correponds to minor axis of curvature (z-axis)
            R = np.hstack((eig_vecs[:,1:2], eig_vecs[:,0:1], eig_vecs[:,2:3]))  # need to reshape vectors to col vectors

            # 6. Check if matrix is improper (is actually both a rotation and reflection), if so, fix it
            if np.linalg.det(R) < 0:  # if det is neg, this means rot matrix is improper
                R[:, 0] *= -1  # multiply left column (v2) by -1 to fix improperness

            #7. Create a rigid transform with the rotation of the normal and position of the point in the PC
            X_OF = RigidTransform(RotationMatrix(R), points[:,index])  # modify here.

            return X_OF
    # def model_pred(self, context, output):
    #     if self.traj_count < 5:
    #         print("collecting traj/pc data")
    #         return
    #     else:
    #         if self.selected_grasp1_world_frame == None:
    #             with torch.no_grad():
    #                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #                 start_time = time.time()
    #                 batch_size = 1
    #                 pointnet_model = PointNetPlusPlus()
    #                 transformer_model = TrajTransformer(feature_size = 12, nhead = 2, num_encoder_layers = 2, num_decoder_layers = 2, dim_feedforward = 512, max_seq_length = 16)
    #                 # traj_mlp_model = TrajMLP(input_size = (150*12), hidden_sizes = [256, 128])
    #                 mlp_model = PredictionMLP(input_size = (512+256), hidden_sizes = [512, 256, 128])
    #                 time_embedding = SinusoidalTimeEmbedding(embedding_dim = 8)

    #                 pointnet_model.to(device)
    #                 transformer_model.to(device)
    #                 # traj_mlp_model.to(device)
    #                 mlp_model.to(device)

    #                 # Load the trained weights
    #                 pointnet_model.load_state_dict(torch.load('model/pointnet_model_weights.pth', map_location=torch.device('cuda')))
    #                 transformer_model.load_state_dict(torch.load('model/transformer_model_weights.pth', map_location=torch.device('cuda')))
    #                 # traj_mlp_model.load_state_dict(torch.load('model/traj_mlp_model_weights.pth', map_location=torch.device('cuda')))
    #                 mlp_model.load_state_dict(torch.load('model/mlp_model_weights.pth', map_location=torch.device('cuda')))
                    
    #                 # Switch all models to evaluation mode
    #                 pointnet_model.eval()
    #                 transformer_model.eval()
    #                 # traj_mlp_model.eval()
    #                 mlp_model.eval()

    #                 traj_pos_scaler_path = 'model/traj_pos_scaler.joblib'
    #                 traj_vel_scaler_path = 'model/traj_vel_scaler.joblib'
    #                 pc_scaler_path = 'model/pc_scaler.joblib'
    #                 X_WG1_scaler_path = 'model/X_WG1_scaler.joblib'
    #                 X_WG2_scaler_path = 'model/X_WG2_scaler.joblib'

    #                 traj_pos_scaler = load(traj_pos_scaler_path)
    #                 traj_vel_scaler = load(traj_vel_scaler_path)
    #                 pc_scaler = load(pc_scaler_path)
    #                 X_WG1_scaler = load(X_WG1_scaler_path)
    #                 X_WG2_scaler = load(X_WG2_scaler_path)

    #                 traj_data = np.array(self.traj_input)
    #                 pc_data = np.array(self.pointcloud_input)
    #                 time_data = np.array(self.time_input)
                    
    #                 traj_pos_data = traj_data[4,:,0:3]
    #                 traj_vel_data = traj_data[4,:,9:12]
    #                 traj_pos_input_normalized = traj_pos_scaler.transform(traj_pos_data.reshape(-1, traj_pos_data.shape[-1])).reshape(traj_pos_data.shape)
    #                 traj_vel_input_normalized = traj_vel_scaler.transform(traj_vel_data.reshape(-1, traj_vel_data.shape[-1])).reshape(traj_vel_data.shape)
    #                 # time_embeddings = time_embedding(torch.tensor(time_data).to(device))
    #                 # time_embeddings_expanded = time_embeddings.unsqueeze(1).repeat(1, 150, 1).to(device)               #5x150x16
    #                 traj_data_normalized = traj_data[4,:,0:12]
    #                 traj_data_normalized[:,0:3] = traj_pos_input_normalized
    #                 traj_data_normalized[:,9:12] = traj_vel_input_normalized
    #                 # traj_tensor = torch.tensor(traj_data_normalized, dtype=torch.float32).to(device)
    #                 traj_with_time_tensor = torch.tensor(traj_data_normalized, dtype=torch.float32).to(device)
    #                 # traj_with_time_tensor= torch.cat((traj_tensor, time_embeddings_expanded), dim=2).to(device)
    #                 print('traj_ori', traj_with_time_tensor[75,:])
    #                 print('traj_ori_2', traj_with_time_tensor[149,:])
    #                 pointcloud_input_normalized = pc_scaler.transform(pc_data.reshape(-1, pc_data.shape[-1])).reshape(pc_data.shape)
                    
    #                 # traj_input_normalized_tensor = torch.tensor(traj_data_with_time, dtype=torch.float32).to('cpu')
    #                 pointcloud_input_normalized_tensor = torch.tensor(pointcloud_input_normalized, dtype=torch.float32).to(device)
    #                 # pointcloud_input_normalized_tensor = pointcloud_input_normalized_tensor.unsqueeze(0)  # Add batch dimension
                    
    #                 # print('shape', pc_data.shape)
    #                 pointnet_input = [Data(pos = pointcloud_input_normalized_tensor[i]) for i in range(pointcloud_input_normalized_tensor.size(0))]
    #                 pointnet_batch = Batch.from_data_list(pointnet_input).to(device)
    #                 # print('batch_shape', pointnet_batch.size())

    #                 timesteps, features = traj_with_time_tensor.size()
    #                 # # Reshape to [batch_size, timesteps * points, features]
    #                 # reshaped_data = traj_with_time_tensor.view(batch_size, timesteps * points, features)  
    #                 # # Transpose to match Transformer's expected input shape [seq_len, batch, features]
    #                 # src_transformer = reshaped_data.transpose(0, 1)
    #                 src_transformer = traj_with_time_tensor.view(batch_size, timesteps, features).transpose(0,1)
                    
    #                 print(f'------------data-preprocessing time:{time.time() - start_time}')
    #                 pointnet_out = pointnet_model(pointnet_batch)
    #                 transformer_out = transformer_model(src = src_transformer, tgt = None)
    #                 print('traj',transformer_out)
    #                 transformer_out_agg = transformer_out.transpose(0, 1).mean(dim=1)
    #                 pointnet_out_agg = pointnet_out.view(batch_size, 5, 512).mean(dim=1)  # Mean pooling over the 5 dimension [batch_size, 1024]
    #                 # transformer_output_agg = transformer_out.view(16, 150, batch_size, 12).mean(dim=1)  # Mean pooling over the 150 dimension [16, batch_size, 16]
    #                 # transformer_output_agg_flat = transformer_output_agg.transpose(0, 1).reshape(batch_size, -1)  # [batch_size, 16*16]
    #                 # traj_flat = traj_with_time_tensor.view(batch_size, 150*12)
    #                 # traj_out = traj_mlp_model(traj_with_time_tensor.view(batch_size, 150*12))
    #                 combined_features = torch.cat((pointnet_out_agg, transformer_out_agg), dim=1)

    #                 # transformer_vis = transformer_out.view(timesteps, features).cpu().detach().numpy()
    #                 # transformer_vis = transformer_out.transpose(0, 1).squeeze().cpu().detach().numpy()
    #                 # print('traj_1', transformer_vis.shape)
    #                 # print('traj_2', transformer_vis[120,:])
    #                 # print('traj_3', transformer_vis[50,:])
    #                 # print('traj_4', transformer_vis[140,:])
    #                 # if self.meshcat is not None:
    #                 #     for t in range(16):
    #                 #         for i in range(150):
    #                 #             self.meshcat.SetObject(f"RansacSpheres/{t}", Sphere(0.02), Rgba(1, 0, 1, 1))
    #                 #             self.meshcat.SetTransform(f"RansacSpheres/{t}", RigidTransform(RotationMatrix(), transformer_vis[t,i,0:3].reshape(3,1)))
    #                             # print(transformer_vis[t,i,0:3])
                    
    #                 xw_1_pred, xw_2_pred, obj_catch_t_pred = mlp_model(combined_features)
    #                 obj_catch_t_pred = obj_catch_t_pred.cpu().detach().numpy() 
    #                 self.obj_catch_t_pred = np.squeeze(obj_catch_t_pred)
    #                 xw_1_pred = xw_1_pred.cpu().detach().numpy()  # Convert to numpy array if they are tensors
    #                 xw_2_pred = xw_2_pred.cpu().detach().numpy()
    #                 xw_1_pred_03 = X_WG1_scaler.inverse_transform(xw_1_pred[:,0:3])
    #                 xw_2_pred_03 = X_WG2_scaler.inverse_transform(xw_2_pred[:,0:3])
    #                 xw_1_pred[:,0:3]  = xw_1_pred_03
    #                 xw_2_pred[:,0:3]  = xw_2_pred_03
    #                 # print('xw1', xw_1_pred)
    #                 # print('xw2', xw_2_pred)
    #                 v1 = xw_1_pred[:,3:6]
    #                 v2 = xw_1_pred[:,6:9]
    #                 v3 = np.cross(v1,v2)
    #                 v3_normalized = v3 / np.linalg.norm(v3)
    #                 rotation_matrix_xw_1 = np.column_stack((v1.reshape(3,1), v2.reshape(3,1), v3_normalized.reshape(3,1)))

    #                 v1 = xw_2_pred[:,3:6]
    #                 v2 = xw_2_pred[:,6:9]
    #                 v3 = np.cross(v1,v2)
    #                 v3_normalized = v3 / np.linalg.norm(v3)
    #                 rotation_matrix_xw_2 = np.column_stack((v1.reshape(3,1), v2.reshape(3,1), v3_normalized.reshape(3,1)))

    #                 self.selected_grasp1_obj_frame = RigidTransform(RotationMatrix(rotation_matrix_xw_1), xw_1_pred[:,0:3].reshape(3,1))
    #                 self.selected_grasp2_obj_frame = RigidTransform(RotationMatrix(rotation_matrix_xw_2), xw_2_pred[:,0:3].reshape(3,1))
    #                 self.draw_grasp_candidate(self.selected_grasp1_obj_frame, self.selected_grasp2_obj_frame, prefix="grippers_best", random_transform=False)
                    

    #                 obj_traj = self.get_input_port(0).Eval(context)
    #                 self.obj_pose_at_catch = obj_traj.value(self.obj_catch_t_pred)
    #                 self.selected_grasp1_world_frame = self.obj_pose_at_catch @ self.selected_grasp1_obj_frame
    #                 self.selected_grasp2_world_frame = self.obj_pose_at_catch @ self.selected_grasp2_obj_frame

    #                 print(f'------------infer time:{time.time() - start_time}')
    #                 print(f'obj_catch_t_pred:{self.obj_catch_t_pred}, X_WG1_predict:{self.selected_grasp1_world_frame}, X_WG2_predict:{self.selected_grasp2_world_frame}')
    #                 output.set_value({'gripper1': (self.selected_grasp1_world_frame, self.obj_catch_t_pred), 'gripper2': (self.selected_grasp2_world_frame, self.obj_catch_t_pred)})
    #         else:
    #             obj_traj = self.get_input_port(0).Eval(context)
                
    #             estimated_obj_catch_pose = RigidTransform(self.obj_pose_at_catch.rotation(), obj_traj.value(self.obj_catch_t_pred).translation())
    #             self.selected_grasp1_world_frame = estimated_obj_catch_pose @ self.selected_grasp1_obj_frame
    #             self.selected_grasp2_world_frame = estimated_obj_catch_pose @ self.selected_grasp2_obj_frame
    #             self.draw_grasp_candidate(self.selected_grasp1_world_frame, self.selected_grasp2_world_frame, prefix="grippers_best1", random_transform=False)
    #             output.set_value({'gripper1': (self.selected_grasp1_world_frame, self.obj_catch_t_pred), 'gripper2': (self.selected_grasp2_world_frame, self.obj_catch_t_pred)})

                    

    # def data_collect(self, context, state):
    #     # grasp = self.get_input_port(2).Eval(context)
    #     # _, self.obj_catch_t = grasp['gripper1']
    #     # if context.get_time() >= self.obj_catch_t - 0.1 and context.get_time() < self.obj_catch_t - 0.05:
    #     #     self.X_WG1, _ = grasp['gripper1']
    #     #     self.X_WG2, _ = grasp['gripper2']
    #         # print(f'X_WG1:{self.X_WG1}, X_WG2:{self.X_WG2}')

    #     if self.traj_count < 5:
    #         traj = self.traj_collect(context)
    #         pointcloud = self.pc_collect(context)
    #         time = context.get_time()

    #         ## visualize pc input
    #         # cloud = PointCloud(pointcloud.T.shape[1])
    #         # if pointcloud.T.shape[1] > 0:
    #         #     cloud.mutable_xyzs()[:] = pointcloud.T
    #         # if self.meshcat is not None:
    #         #     print(f'pc drew')
    #         #     self.meshcat.SetObject(f"{str(self)}PointCloud{self.traj_count}", cloud, point_size=0.01, rgba=Rgba(0, 1.0, 0.5))

    #         self.traj_input.append(traj)    #5x150x15
    #         self.pointcloud_input.append(pointcloud)    #5x1024x3
    #         self.time_input.append(time)    #5x1    

    #         self.traj_count += 1
    #     elif self.traj_count >= 5 and self.traj_count <= 20:
    #         # self.model_pred()
    #         traj_after = self.traj_collect(context)
    #         time_after = context.get_time()

    #         self.traj_input_after.append(traj_after)    #16x150x15
    #         self.time_input_after.append(time_after)    #16x1 

    #         self.traj_count += 1
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