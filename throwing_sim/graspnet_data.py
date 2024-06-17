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
import numpy as np
import h5py
from utils import ObjectTrajectory
from joblib import load
import torch
from torch_geometric.data import Data, Batch
from scipy.spatial import cKDTree as KDTree

from graspnet_11 import PointNetPlusPlus, TrajTransformer, PredictionMLP


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
    def __init__(self, original_plant, scene_graph, model_name, grasp_random_seed, velocity, roll, launching_position, launching_orientation, initial_pose, meshcat):
        LeafSystem.__init__(self)

        self.plant = original_plant
        self.graph = scene_graph
        self.model_name = model_name
        self.grasp_random_seed = grasp_random_seed
        self.random_transform = RigidTransform([-1, -1, 1])
        
        obj_traj = AbstractValue.Make(ObjectTrajectory())
        self.DeclareAbstractInputPort("object_trajectory", obj_traj)    #0 input
        obj_pc = AbstractValue.Make(PointCloud())
        self.DeclareAbstractInputPort("object_pc", obj_pc)              #1 input
        grasp = AbstractValue.Make({'gripper1': (RigidTransform(), 0), 'gripper2': (RigidTransform(), 0), 'gripper1_pc':(RigidTransform(), 0), 'gripper2_pc':(RigidTransform(), 0), 'index1': (0), 'index2': (0)}) # right:gripper1, left:gripper2
        self.DeclareAbstractInputPort("grasp_selection", grasp)         #2 input
        self.DeclareAbstractInputPort("contact_results_input", 
            AbstractValue.Make(ContactResults())                        #3 input
        )
        state = AbstractValue.Make((RigidTransform(), 0))
        self.DeclareAbstractInputPort("noodle_state", state)  # [x, y, z, roll, pitch, yaw]  #4 input


        self.traj_count = 0
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.025, 0.0, self.data_collect)

       

        self.traj_input = []
        self.pointcloud_input = []
        self.pointcloud_normal = []
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
        self.initial_pose = initial_pose

        self.meshcat = meshcat
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
        target_num_points = 1700 #4056
        pc = self.get_input_port(1).Eval(context).VoxelizedDownSample(voxel_size=0.0025)
        pc.EstimateNormals(0.05, 30)
        pc_normals = pc.normals().T
        pc = pc.xyzs().T      #Nx3

        num_points = pc.shape[0]
        print(f'num_points:{num_points}')
        print(f'normal_shape:{pc_normals.shape}')
        if num_points == target_num_points:
            return pc, pc_normals
        elif num_points > target_num_points:
            # Randomly select 'target_num_points' from the point cloud
            indices = np.random.choice(num_points, target_num_points, replace=False)
            pc = pc[indices, :]
            pc_normals = pc_normals[indices, :]
        else:
            # If fewer points than needed, duplicate some points
            indices = np.random.choice(num_points, target_num_points, replace=True)
            pc = pc[indices, :]
            pc_normals = pc_normals[indices, :]
        # print('pc', pc.shape)  
        # kdtree = KDTree(pc)
        # _, index_1 = kdtree.query(first_grasp_point.reshape(1, -1), k=1)
        # _, index_2 = kdtree.query(second_grasp_point.reshape(1, -1), k=1)   
        # grasp1 = pc[index_1,:].T
        # grasp2 = pc[index_2,:].T
        pc_combined = np.concatenate([pc, pc_normals], axis = 1)
        print('concat',pc_combined.shape)
        return pc, pc_normals 


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
        # print(f'time:{context.get_time()}, catch_t:{self.obj_catch_t}')
        # if context.get_time() >= self.obj_catch_t - 0.1 and context.get_time() < self.obj_catch_t - 0.05:

        # position = X_WB.translation()
        # orientation = X_WB.rotation().ToQuaternion()
        # print(f'pos:{position}, ori:{orientation}')
        if context.get_time() >= self.obj_catch_t - 0.1 and context.get_time() < self.obj_catch_t - 0.05:
            self.X_WG1, _ = grasp['gripper1']
            self.X_WG2, _ = grasp['gripper2']
            # self.draw_grasp_candidate(self.X_WG1, self.X_WG2, prefix="gripper1and2", random_transform=True)

        if self.traj_count < 5:
            self.traj = self.traj_collect(context)
            pointcloud, normals = self.pc_collect(context)
            self.trans_pc = pointcloud
            # trans_normals = (self.initial_pose @ normals.T).T
            # print('normal',trans_normals)
            self.trans_normals = (self.initial_pose.rotation() @ normals.T).T
            # print('normal',trans_normals)
            # self.pointcloud = np.concatenate([trans_pc, trans_normals], axis = 1)
            # print(f'pc shape:{self.pointcloud.shape}')
            time = context.get_time()
            grasp = self.get_input_port(2).Eval(context)
            _, self.obj_catch_t = grasp['gripper1']
            self.X_OG1 = grasp['gripper1_pc']
            self.X_OG2 = grasp['gripper2_pc']
            if self.traj_count == 4:
                rotation = RollPitchYaw(np.radians(90), 0, 0).ToRotationMatrix()
                # Apply this rotation to the rotation part of the existing RigidTransform
                new_rotation = rotation @ self.X_OG1.rotation()
                # Create a new RigidTransform with the new rotation and the old translation
                self.X_OG1 = RigidTransform(new_rotation, rotation @ self.X_OG1.translation())
                new_rotation_2 = rotation @ self.X_OG2.rotation()
                # Create a new RigidTransform with the new rotation and the old translation
                self.X_OG2 = RigidTransform(new_rotation_2, rotation @ self.X_OG2.translation())
                print(f'X_OG1:{self.X_OG1}, X_OG2:{self.X_OG2}')
                self.draw_grasp_candidate(self.X_OG1, self.X_OG2, prefix="grippers_best_trans", random_transform=False)
            # self.index_1 = grasp['index1']
            # self.index_2 = grasp['index2']
            # print(f'index_1:{self.index_1}, index_2:{self.index_2}')
            # X_WB = self.get_input_port(4).Eval(context)
            # print(f'X_WB{X_WB}')

            
            
            # self.point_1 = (X_WB @ self.X_WG1) #right gripper
            # self.point_2 = (X_WB @ self.X_WG2) #left
            # self.pointcloud, self.index_1, self.index_2, self.grasp1, self.grasp2 = self.pc_collect(context, self.point_1, self.point_2)
            # self.points = np.concatenate((self.point_1.reshape(1,3), self.point_2.reshape(1,3)) , axis = 0)
            # print(f'point_1:{self.point_1} point_2:{self.point_2} points:{self.points.shape}')
            # print(f'index_1:{self.index_1} index_2:{self.index_2} grasp1:{self.grasp1} grasp2:{self.grasp2}')
            # visualize pc input
            cloud = PointCloud(self.trans_pc.T.shape[1])
            if self.trans_pc.T.shape[1] > 0:
                cloud.mutable_xyzs()[:] = self.trans_pc.T
            if self.meshcat is not None:
                print(f'pc drew')
                self.meshcat.SetObject(f"{str(self)}PointCloud{self.traj_count}", cloud, point_size=0.01, rgba=Rgba(0, 1.0, 0.5))
                # self.meshcat.SetObject(f"point/1", Sphere(0.005), Rgba(0.2, 0.2, 1, 1))
                # self.meshcat.SetTransform(f"point/1", RigidTransform(self.point_1))
                # self.meshcat.SetObject(f"point/2", Sphere(0.005), Rgba(0.2, 0.2, 1, 1))
                # self.meshcat.SetTransform(f"point/2", RigidTransform(self.point_2))
                # self.meshcat.SetObject(f"point/1pc", Sphere(0.005), Rgba(0.2, 0.2, 1, 1))
                # self.meshcat.SetTransform(f"point/1pc", RigidTransform(self.X_WG1))
                # self.meshcat.SetObject(f"point/2pc", Sphere(0.005), Rgba(0.2, 0.2, 1, 1))
                # self.meshcat.SetTransform(f"point/2pc", RigidTransform(self.X_WG2))

                # self.meshcat.SetObject(f"grasp1/1pc", Sphere(0.0025), Rgba(1.0, 0.2, 1, 1))
                # self.meshcat.SetTransform(f"grasp1/1pc", RigidTransform(self.grasp1))
                # self.meshcat.SetObject(f"grasp2/2pc", Sphere(0.0025), Rgba(1.0, 0.2, 1, 1))
                # self.meshcat.SetTransform(f"grasp2/2pc", RigidTransform(self.grasp2))
            # self.grasp1 = self.grasp1.reshape(1,3)
            # self.grasp2 = self.grasp2.reshape(1,3)

            # self.traj_input.append(traj)    #5x150x15
            self.pointcloud_input.append(pointcloud)    #5x1024x3
            self.pointcloud_normal.append(normals)
            # self.time_input.append(time)    #5x1    

            self.traj_count += 1
            # obj_traj = self.get_input_port(0).Eval(context)
            # self.obj_pose_at_catch = obj_traj.value(self.obj_catch_t).translation()  #0.1s
            # print(f'{self.obj_pose_at_catch}, count:{self.traj_count}')

        # elif self.traj_count >= 5 and self.traj_count <= 20:
        #     # self.model_pred()
        #     traj_after = self.traj_collect(context)
        #     time_after = context.get_time()

        #     self.traj_input_after.append(traj_after)    #16x150x15
        #     self.time_input_after.append(time_after)    #16x1 

        #     self.traj_count += 1
        else:
            if context.get_time() >= 0.8 and self.write:
                self.write = False 
                result = 1 if self.check_success(context) else 0
                #TODO write traj,pointcloud,time as input/ X_WG1,X_WG2,obj_catch_t as output in h5py
                X_WG1_rot = self.X_WG1.rotation().matrix()
                X_WG1_rotation = np.concatenate((X_WG1_rot[:,0].reshape(3,1),X_WG1_rot[:,1].reshape(3,1),X_WG1_rot[:,2].reshape(3,1)), axis = None)  #6d rotation representation
                X_WG1_translation = self.X_WG1.translation()   #3x1
                # X_WG1_input = np.concatenate((X_WG1_translation, X_WG1_rotation), axis = None)  #9x1

                X_WG2_rot = self.X_WG2.rotation().matrix()
                X_WG2_rotation = np.concatenate((X_WG2_rot[:,0].reshape(3,1),X_WG2_rot[:,1].reshape(3,1),X_WG2_rot[:,2].reshape(3,1)), axis = None)  #6d rotation representation
                X_WG2_translation = self.X_WG2.translation()   #3x1
                # X_WG2_input = np.concatenate((X_WG2_translation, X_WG2_rotation), axis = None)  #9x1

                X_OG1_rot = self.X_OG1.rotation().matrix()
                X_OG1_rotation = np.concatenate((X_OG1_rot[:,0].reshape(3,1),X_OG1_rot[:,1].reshape(3,1),X_OG1_rot[:,2].reshape(3,1)), axis = None)  #6d rotation representation
                X_OG1_translation = self.X_OG1.translation()   #3x1
                # X_OG1_input = np.concatenate((X_OG1_translation, X_OG1_rotation), axis = None)  #9x1

                X_OG2_rot = self.X_OG2.rotation().matrix()
                X_OG2_rotation = np.concatenate((X_OG2_rot[:,0].reshape(3,1),X_OG2_rot[:,1].reshape(3,1),X_OG2_rot[:,2].reshape(3,1)), axis = None)  #6d rotation representation
                X_OG2_translation = self.X_OG2.translation()   #3x1
                # X_OG2_input = np.concatenate((X_OG2_translation, X_OG2_rotation), axis = None)  #9x1

                initial_rot = self.initial_pose.rotation().matrix()
                initial_rotation = np.concatenate((initial_rot[:,0].reshape(3,1),initial_rot[:,1].reshape(3,1),initial_rot[:,2].reshape(3,1)), axis = None)
                initial_translation = self.initial_pose.translation()   #3x1
                initial_pos = np.concatenate((initial_translation, initial_rotation), axis = None)
                self.write_data_to_h5py(self.traj, self.pointcloud_input, self.pointcloud_normal , X_WG1_rotation, X_WG1_translation, X_WG2_rotation, X_WG2_translation, X_OG1_rotation, X_OG1_translation,\
                X_OG2_rotation, X_OG2_translation, result, initial_pos)
                self.clear_data_buffers()

    
    def write_data_to_h5py(self, traj, pointcloud, normals, X_WG1_rotation, X_WG1_translation, X_WG2_rotation, X_WG2_translation, X_OG1_rotation, X_OG1_translation,\
                X_OG2_rotation, X_OG2_translation, result, initial_pos):
        if result == 1:
        # Create or open an HDF5 file
            with h5py.File('graspnet9_data_5pc_matrix.h5', 'a') as hf:
                # Check if datasets exist, and get the new index
                index = len(hf.keys()) // 14
                #input
                hf.create_dataset(f'traj_data_{index}', data=np.array(traj))     #5x150x15
                hf.create_dataset(f'pc_data_{index}', data=np.array(pointcloud)) #5x1024x3
                hf.create_dataset(f'pc_normal_{index}', data=np.array(normals)) #5x1024x3
                # hf.create_dataset(f'time_data_{index}', data=np.array(self.time_input))     #5x1 
                # hf.create_dataset(f'traj_data_after_{index}', data=np.array(self.traj_input_after))     #16x150x15
                # hf.create_dataset(f'time_data_after_{index}', data=np.array(self.time_input_after))     #16x1 
                #output
                hf.create_dataset(f'X_WG1_rot_{index}', data=np.array(X_WG1_rotation))             #9x1
                hf.create_dataset(f'X_WG1_tran_{index}', data=np.array(X_WG1_translation))             #9x1
                hf.create_dataset(f'X_WG2_rot_{index}', data=np.array(X_WG2_rotation))             #9x1
                hf.create_dataset(f'X_WG2_tran_{index}', data=np.array(X_WG2_translation))             #9x1
                hf.create_dataset(f'X_OG1_rot_{index}', data=np.array(X_OG1_rotation))             #9x1
                hf.create_dataset(f'X_OG1_tran_{index}', data=np.array(X_OG1_translation))             #9x1
                hf.create_dataset(f'X_OG2_rot_{index}', data=np.array(X_OG2_rotation))             #9x1
                hf.create_dataset(f'X_OG2_tran_{index}', data=np.array(X_OG2_translation))             #9x1
                hf.create_dataset(f'obj_catch_t_{index}', data=np.array(self.obj_catch_t))        #1
                hf.create_dataset(f'result_{index}', data=np.array(result))                  #1
                hf.create_dataset(f'initial_pose_{index}', data=np.array(initial_pos))                  #1
                # hf.create_dataset(f'point1_{index}', data=np.array(point1))  
                # hf.create_dataset(f'point2_{index}', data=np.array(point2)) 
                # hf.create_dataset(f'index1_{index}', data=np.array(index_1))   
                # hf.create_dataset(f'index2_{index}', data=np.array(index_2))
            
            # with h5py.File('graspnet_data_addtraj_bar.h5', 'a') as hf:
            #     # Check if datasets exist, and get the new index
            #     index = len(hf.keys()) // 9
            #     #input
            #     hf.create_dataset(f'traj_data_{index}', data=np.array(self.traj_input))     #5x150x15
            #     hf.create_dataset(f'pc_data_{index}', data=np.array(self.pointcloud_input)) #5x1024x3
            #     hf.create_dataset(f'time_data_{index}', data=np.array(self.time_input))     #5x1 
            #     hf.create_dataset(f'traj_data_after_{index}', data=np.array(self.traj_input_after))     #16x150x15
            #     hf.create_dataset(f'time_data_after_{index}', data=np.array(self.time_input_after))     #16x1 
            #     #output
            #     hf.create_dataset(f'X_WG1_{index}', data=np.array(X_WG1_input))             #9x1
            #     hf.create_dataset(f'X_WG2_{index}', data=np.array(X_WG2_input))             #9x1
            #     hf.create_dataset(f'obj_catch_t_{index}', data=np.array(self.obj_catch_t))        #1
            #     hf.create_dataset(f'result_{index}', data=np.array(result))                  #1

        with open('graspnet9_data_result_5pc_matrix.txt', "a") as text_file:
            # Write the information to the file
            text_file.write(f"Object: {self.model_name}, Seed: {self.grasp_random_seed}, Result: {result}, vel: {self.velocity}, pos:{self.launching_position}, ori:{self.launching_orientation}\n")
            # text_file.write(f"Traj: {self.traj_input[4]}\n")
            # text_file.write(f"catch: {self.obj_pose_at_catch}\n")
            # text_file.write(f"Traj_after: {self.traj_input_after}\n")
    
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
                
                    

