from pydrake.all import (
    AbstractValue,
    Concatenate,
    LeafSystem,
    PointCloud,
    AddMultibodyPlantSceneGraph,
    Box,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Quaternion,
    InverseKinematics,
    Solve,
    Cylinder,
    RollPitchYaw
)
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddMultibodyTriad
from manipulation.utils import ConfigureParser

import time
import numpy as np
from scipy.spatial import cKDTree as KDTree

from utils import ObjectTrajectory


class GraspSelector(LeafSystem):
    def __init__(self, plant, scene_graph, meshcat, thrown_model_name, grasp_random_seed, iiwa1_pose, iiwa2_pose):
        LeafSystem.__init__(self)
        obj_pc = AbstractValue.Make(PointCloud())
        obj_traj = AbstractValue.Make(ObjectTrajectory())
        self.DeclareAbstractInputPort("object_pc", obj_pc)
        self.DeclareAbstractInputPort("object_trajectory", obj_traj)

        port = self.DeclareAbstractOutputPort(
            "grasp_selection",
            lambda: AbstractValue.Make({'gripper1': (RigidTransform(), 0), 'gripper2': (RigidTransform(), 0)}),  # dict mapping grasp to a grasp time
            self.SelectGrasp,
        )
        port.disable_caching_by_default()

        self._rng = np.random.default_rng()
        self.plant = plant
        self.scene_graph = scene_graph
        self.meshcat = meshcat
        self.thrown_model_name = thrown_model_name
        self.grasp_random_seed = grasp_random_seed
        self.iiwa1_pose = iiwa1_pose
        self.iiwa2_pose = iiwa2_pose

        self.random_transform = RigidTransform([-1, -1, 1])  # used for visualizing grasp candidates off to the side
        self.selected_grasp1_obj_frame = None
        self.selected_grasp2_obj_frame = None
        self.obj_catch_t = None
        self.visualize = True


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


    def check_collision(self, obj_pc, X_G1, X_G2):
        """
        Builds a new MBP and diagram with just the object and WSG, and computes
        SDF to check if there is collision.
        """
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        ConfigureParser(parser)
        gripper_model_url = "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
        gripper_model_url2 = "package://manipulation/schunk_wsg_50_welded_fingers_copy.sdf"
        url = '/home/haonan/Catching_bot/throwing_sim/schunk_wsg_50_welded_fingers2.sdf'
        gripper1_instance = parser.AddModelsFromUrl(gripper_model_url)[0]
        # gripper2_instance = parser.AddModelsFromUrl(gripper_model_url2)[0]
        gripper2_instance = parser.AddModels(url)[0]

        AddMultibodyTriad(plant.GetFrameByName("body", gripper1_instance), scene_graph)
        AddMultibodyTriad(plant.GetFrameByName("body", gripper2_instance), scene_graph)
        plant.Finalize()

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        plant_context = plant.GetMyContextFromRoot(context)
        scene_graph_context = scene_graph.GetMyContextFromRoot(context)
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body", gripper1_instance), X_G1)
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body", gripper2_instance), X_G2)

        query_object = scene_graph.get_query_output_port().Eval(
            scene_graph_context
        )

        for pt in obj_pc.xyzs().T:
            distances = query_object.ComputeSignedDistanceToPoint(pt)
            for body_index in range(len(distances)):
                distance = distances[body_index].distance
                if distance < 0:
                    # print('collision true')
                    return True  # Collision
        # print('collision false')
        return False
    

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
    

    def check_nonempty(self, obj_pc, X_WG1, X_WG2, visualize=False):
        """
        Check if the "closing region" of the gripper is nonempty by transforming the
        pointclouds to gripper coordinates.

        Args:
        - obj_pc (PointCloud object): pointcloud of the object.
        - X_WG (Drake RigidTransform): transform of the gripper.
        Return:
        - is_nonempty (boolean): boolean set to True if there is a point within
            the cropped region.
        """
        obj_pc_W_np = obj_pc.xyzs()

        # Bounding box of the closing region written in the coordinate frame of the gripper body.
        # Do not modify
        crop_min = [-0.05, 0.05, -0.00625]
        crop_max = [0.05, 0.1125, 0.00625]

        # Transform the pointcloud to gripper frame.
        X_GW1 = X_WG1.inverse()
        obj_pc_G1_np = X_GW1.multiply(obj_pc_W_np)
        X_GW2 = X_WG2.inverse()
        obj_pc_G2_np = X_GW2.multiply(obj_pc_W_np)
        # Check if there are any points within the cropped region.

        indices1 = np.all(
            (
                crop_min[0] <= obj_pc_G1_np[0, :],
                obj_pc_G1_np[0, :] <= crop_max[0],
                crop_min[1] <= obj_pc_G1_np[1, :],
                obj_pc_G1_np[1, :] <= crop_max[1],
                crop_min[2] <= obj_pc_G1_np[2, :],
                obj_pc_G1_np[2, :] <= crop_max[2],
            ),
            axis=0,
        )

        is_nonempty1 = indices1.any()

        indices2 = np.all(
            (
                crop_min[0] <= obj_pc_G2_np[0, :],
                obj_pc_G2_np[0, :] <= crop_max[0],
                crop_min[1] <= obj_pc_G2_np[1, :],
                obj_pc_G2_np[1, :] <= crop_max[1],
                crop_min[2] <= obj_pc_G2_np[2, :],
                obj_pc_G2_np[2, :] <= crop_max[2],
            ),
            axis=0,
        )

        is_nonempty2 = indices2.any()

        if visualize:
            self.meshcat.Delete()
            obj_pc_G1 = PointCloud(obj_pc)
            obj_pc_G1.mutable_xyzs()[:] = obj_pc_G1_np #obj_pc_W_np ??
            obj_pc_G2 = PointCloud(obj_pc)
            obj_pc_G2.mutable_xyzs()[:] = obj_pc_G2_np

            self.draw_grasp_candidate(RigidTransform())
            self.meshcat.SetObject("cloud1", obj_pc_G1)
            self.meshcat.SetObject("cloud2", obj_pc_G2)

            box_length = np.array(crop_max) - np.array(crop_min)
            box_center = (np.array(crop_max) + np.array(crop_min)) / 2.0
            # box_center_world = X_WG1.multiply(box_center)  # For gripper 1's closing region ??
            self.meshcat.SetObject(
                "closing_region",
                Box(box_length[0], box_length[1], box_length[2]),
                Rgba(1, 0, 0, 0.3),
            )
            self.meshcat.SetTransform("closing_region", RigidTransform(box_center))

        return is_nonempty1 and is_nonempty2
    

    def compute_grasp_cost(self, point_on_ring, X_OG, t, name):
        """
        Defines cost function that is used to pick best grasp sample.

        Args:
            obj_pc_centroid: (3,) np array
            X_OG: RigidTransform containing gripper pose for this grasp in obj frame
            t: float, time at which the grasp occurs
        """
        if name == 'gripper1':
            # Compute distance from Y-axis ray of gripper frame to objects' left 1/4 point.
            # L = 0.9
            # first_pos = [0, 0, -L/4]
            # obj_pc_1_4_relative_to_X_OG = first_pos - X_OG.translation()
            # X_OG_y_axis_vector = X_OG.rotation().matrix()[:, 1]
            # projection_obj_pc_1_4_relative_to_X_OG_onto_X_OG_y_axis_vector = (np.dot(obj_pc_1_4_relative_to_X_OG, X_OG_y_axis_vector) / np.linalg.norm(X_OG_y_axis_vector)) * X_OG_y_axis_vector  # Equation for projection of one vector onto another
            # # On the order of 0 - 0.05
            # distance_obj_pc_desired_to_X_OG_y_axis = np.linalg.norm(obj_pc_1_4_relative_to_X_OG - projection_obj_pc_1_4_relative_to_X_OG_onto_X_OG_y_axis_vector)

            # Transform the grasp pose from object frame to world frame
            X_WO = self.obj_pose_at_catch
            X_WG = X_WO @ X_OG

            #check alignment with tangent line
            slope = -point_on_ring[1]/point_on_ring[2]
            z = slope * (1 - point_on_ring[1]) + point_on_ring[2]
            ring_A = np.array([0, 1, z])
            ring_B = np.array([0, point_on_ring[1], point_on_ring[2]])
            tangent_line = (ring_A - ring_B)/np.linalg.norm(ring_A - ring_B)
            X_WG_z_axis_vector = (X_WG.rotation().matrix() @ np.array([[0],[0],[1]])).reshape((3,))
            z_alignment = 1 - np.abs(np.dot(tangent_line, X_WG_z_axis_vector))

            # Add cost associated with whether X_WG's y-axis points away from iiwa (which is what we want)
            world_z_axis_to_X_WG_vector = np.append(X_WG.translation()[:2], 0)  # basically replacing z with 0
            world_z_axis_to_iiwa1_base = np.append(self.iiwa1_pose.translation()[:2], 0)
            iiwa_base_to_X_WG_vector = (world_z_axis_to_X_WG_vector - world_z_axis_to_iiwa1_base) / np.linalg.norm(world_z_axis_to_X_WG_vector - world_z_axis_to_iiwa1_base)
            X_WG_y_axis_vector = (X_WG.rotation().matrix() @ np.array([[0],[1],[0]])).reshape((3,))
            # # On the order of 0 - 2
            direction = 1 - np.dot(iiwa_base_to_X_WG_vector, X_WG_y_axis_vector)

            # Add cost associated with whether object is able to fly in between two fingers of gripper
            # Z-axis of gripper should be aligned with derivative of obj trajectory    z-axis should be orthognal to the derivative
            obj_vel_at_catch = self.obj_traj.EvalDerivative(t)[:3]  # (3,) np array
            obj_direction_at_catch = obj_vel_at_catch / np.linalg.norm(obj_vel_at_catch) * -1 # normalize
            # X_WG_z_axis_vector = (X_WG.rotation().matrix() @ np.array([[0],[0],[1]])).reshape((3,))
            # on the order of 0 - 2
            alignment = 1 - np.dot(obj_direction_at_catch, X_WG_y_axis_vector)  # absolute since it's ok for gripper z-axis to be perfectly against obj velocity

            z_axis_penalty = 0 if X_WG_z_axis_vector[2] > 0 else 1 - X_WG_z_axis_vector[2]
        else:
            # L = 0.9
            # second_pos = [0, 0, L/4]
            # obj_pc_3_4_relative_to_X_OG = second_pos - X_OG.translation()
            # X_OG_y_axis_vector = X_OG.rotation().matrix()[:, 1]
            # projection_obj_pc_3_4_relative_to_X_OG_onto_X_OG_y_axis_vector = (np.dot(obj_pc_3_4_relative_to_X_OG, X_OG_y_axis_vector) / np.linalg.norm(X_OG_y_axis_vector)) * X_OG_y_axis_vector  # Equation for projection of one vector onto another
            # # On the order of 0 - 0.05
            # distance_obj_pc_desired_to_X_OG_y_axis = np.linalg.norm(obj_pc_3_4_relative_to_X_OG - projection_obj_pc_3_4_relative_to_X_OG_onto_X_OG_y_axis_vector)

            # Transform the grasp pose from object frame to world frame
            X_WO = self.obj_pose_at_catch
            X_WG = X_WO @ X_OG

            #check alignment with tangent line
            slope = -point_on_ring[1]/point_on_ring[2]
            z = slope * (1 - point_on_ring[1]) + point_on_ring[2]
            ring_A = np.array([0, 1, z])
            ring_B = np.array([0, point_on_ring[1], point_on_ring[2]])
            tangent_line = (ring_A - ring_B)/np.linalg.norm(ring_A - ring_B)
            X_WG_z_axis_vector = (X_WG.rotation().matrix() @ np.array([[0],[0],[1]])).reshape((3,))
            z_alignment = 1 - np.abs(np.dot(tangent_line, X_WG_z_axis_vector))

            # Add cost associated with whether X_WG's y-axis points away from iiwa (which is what we want)
            world_z_axis_to_X_WG_vector = np.append(X_WG.translation()[:2], 0)  # basically replacing z with 0
            world_z_axis_to_iiwa2_base = np.append(self.iiwa2_pose.translation()[:2], 0)
            iiwa_base_to_X_WG_vector = (world_z_axis_to_X_WG_vector - world_z_axis_to_iiwa2_base) / np.linalg.norm(world_z_axis_to_X_WG_vector - world_z_axis_to_iiwa2_base)
            X_WG_y_axis_vector = (X_WG.rotation().matrix() @ np.array([[0],[1],[0]])).reshape((3,))
            # # On the order of 0 - 2
            direction = 1 - np.dot(iiwa_base_to_X_WG_vector, X_WG_y_axis_vector)

            # Add cost associated with whether object is able to fly in between two fingers of gripper
            # Z-axis of gripper should be aligned with derivative of obj trajectory    z-axis should be orthognal to the derivative
            obj_vel_at_catch = self.obj_traj.EvalDerivative(t)[:3]  # (3,) np array
            obj_direction_at_catch = obj_vel_at_catch / np.linalg.norm(obj_vel_at_catch) * -1 # normalize

            # X_WG_z_axis_vector = (X_WG.rotation().matrix() @ np.array([[0],[0],[1]])).reshape((3,))
            # on the order of 0 - 2
            alignment = 1 - np.dot(obj_direction_at_catch, X_WG_y_axis_vector)

            z_axis_penalty = 0 if X_WG_z_axis_vector[2] > 0 else 1 - X_WG_z_axis_vector[2]
        # if (alignment < 0.010 and direction < 0.040):
        #     print(f"\nworld_z_axis_to_X_WG_vector: {world_z_axis_to_X_WG_vector}")
        #     print(f"X_WG_y_axis_vector: {X_WG_y_axis_vector}")
        #     print(f"direction: {direction}")
        #     print(f"\nobj_direction_at_catch: {obj_direction_at_catch}")
        #     print(f"X_WG_z_axis_vector: {X_WG_z_axis_vector}")
        #     print(f"alignment: {alignment}\n")

        # Weight the different parts of the cost function
        final_cost = 10*alignment + direction + 5*z_alignment + 5*z_axis_penalty #+ 10*distance_obj_pc_desired_to_X_OG_y_axis

        return final_cost,  direction, alignment


    def compute_candidate_grasps(self, obj_pc, obj_pc_centroid, obj_catch_t, candidate_num=int(np.random.uniform(100,500))):
        """
        Args:
            - obj_pc (PointCloud object): pointcloud of the object.
            - candidate_num (int) : number of desired candidates.
        Return:
            - candidate_lst (list of drake RigidTransforms) : candidate list of grasps.
        """

        # Constants for random variation
        np.random.seed(self.grasp_random_seed)

        # Build KD tree for the pointcloud.
        kdtree = KDTree(obj_pc.xyzs().T)
        ball_radius = 0.002

        candidate_lst_1 = {}  # dict mapping candidates (given by RigidTransforms) to cost of that candidate
        candidate_lst_2 = {}

        def compute_candidate(center, direction_axis, points,  obj_pc, kdtree, ball_radius, candidate_lst_lock, candidate_lst_1, candidate_lst_2):
            magnitude = np.random.uniform(0.35, 0.5)
            first_grasp_point = center - direction_axis * magnitude
            second_grasp_point = center + direction_axis * magnitude
            # print(f'first:{first_grasp_point}, second:{second_grasp_point}')
            _, index_1 = kdtree.query(first_grasp_point.reshape(1, -1), k=1)
            _, index_2 = kdtree.query(second_grasp_point.reshape(1, -1), k=1)
            original_point_1 = points[:, index_1].reshape(-1, 3)  # Ensure it's 2D array for kdtree.query
            original_point_2 = points[:, index_2].reshape(-1, 3)
            indices_1 = kdtree.query_ball_point(original_point_1[0], 0.05)
            indices_2 = kdtree.query_ball_point(original_point_2[0], 0.05)
            # print(f'first:{original_point_1}, second:{original_point_2}')
            nearby_index_1 = np.random.choice([idx for idx in indices_1 if idx != index_1]) if len(indices_1) > 1 else index_1
            nearby_index_2 = np.random.choice([idx for idx in indices_2 if idx != index_2]) if len(indices_2) > 1 else index_2
            point_on_ring1 = points[:,nearby_index_1]
            point_on_ring2 = points[:,nearby_index_2]
            # print('ring',point_on_ring1)
            X_OF_1 = self.compute_darboux_frame(nearby_index_1, obj_pc, kdtree, ball_radius)  # find Darboux frame of random point
            X_OF_2 = self.compute_darboux_frame(nearby_index_2, obj_pc, kdtree, ball_radius)
            # offset gripper pose from object centroid depending on object size
            # if "banana" in self.thrown_model_name.lower():
            #     y_offset = -0.04
            # if "ball" in self.thrown_model_name.lower():
            #     y_offset = -0.05
            # if "bottle" in self.thrown_model_name.lower():
            #     y_offset = -0.05
            y_offset = -0.06
            new_X_OG_1 = X_OF_1 @ RigidTransform(np.array([0, y_offset, 0]))  # Move gripper back by fixed amount
            new_X_OG_2 = X_OF_2 @ RigidTransform(np.array([0, y_offset, 0]))

            grasp_CoM_cost_threshold = 0.1 #0.030  # range: 0 - 0.05
            direction_cost_threshold = 0.400  # range: 0 - 2
            collision_cost_threshold = 0.100  # range: 0 - 2
            new_X_OG_cost_1,  direction_cost_1, collision_cost_1 = self.compute_grasp_cost(point_on_ring1, new_X_OG_1, obj_catch_t, 'gripper1')
            new_X_OG_cost_2, direction_cost_2, collision_cost_2 = self.compute_grasp_cost(point_on_ring2, new_X_OG_2, obj_catch_t, 'gripper2')
            # if grasp isn't above thresholds, don't even bother checking for collision (which is slow)
            # if grasp_CoM_cost_1 > grasp_CoM_cost_threshold or direction_cost_1 > direction_cost_threshold or collision_cost_1 > collision_cost_threshold:
            #     return
            # if grasp_CoM_cost_2 > grasp_CoM_cost_threshold or direction_cost_2 > direction_cost_threshold or collision_cost_2 > collision_cost_threshold:
            #     return

            # print("passed grasping thresholds")

            # check_collision takes most of the runtime
            if (self.check_collision(obj_pc, new_X_OG_1, new_X_OG_2) is not True) and self.check_nonempty(obj_pc, new_X_OG_1, new_X_OG_2):  # no collision, and there is an object between fingers
                with candidate_lst_lock:
                    # print('adding candidate')
                    candidate_lst_1[new_X_OG_1] = new_X_OG_cost_1
                    candidate_lst_2[new_X_OG_2] = new_X_OG_cost_2

        import threading

        threads = []
        candidate_lst_lock = threading.Lock()
        points = obj_pc.xyzs()
        direction_axis = np.array([0,-1,0]) #self.obj_pose_at_catch.rotation().matrix()[:,2] #self.obj_traj.value(0).rotation().matrix()[:,2] #find obj current pose's y-axis (in world frame)
        # print(self.obj_traj.value(0).rotation().matrix())
        # print(self.obj_traj.value(0.3).rotation().matrix())
        for _ in range(candidate_num):
            center = obj_pc_centroid
            t = threading.Thread(target=compute_candidate, args=(center,
                                                                direction_axis,
                                                                points,
                                                                obj_pc,
                                                                kdtree,
                                                                ball_radius,
                                                                candidate_lst_lock,
                                                                candidate_lst_1,
                                                                candidate_lst_2))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if len(candidate_lst_1) == 0 or len(candidate_lst_2) == 0:
            print("grasp sampling did not find any valid candidates.")

        print(f"found {len(candidate_lst_1)} potential grasps")

        return candidate_lst_1, candidate_lst_2
    


    def SelectGrasp(self, context, output):
        if self.selected_grasp1_obj_frame is None and self.selected_grasp2_obj_frame is None:
            self.obj_pc = self.get_input_port(0).Eval(context).VoxelizedDownSample(voxel_size=0.0025)
            self.obj_pc.EstimateNormals(0.05, 30)  # allows us to use obj_pc.normals() function later
            self.obj_traj = self.get_input_port(1).Eval(context)

            if (self.obj_traj == ObjectTrajectory()):  # default output of TrajectoryPredictor system; means that it hasn't seen the object yet
                print("received default traj (in SelectGrasp)")
                return

            self.meshcat.SetObject("cloud", self.obj_pc)

            obj_pc_centroid = np.mean(self.obj_pc.xyzs(), axis=1)  # column-wise mean of 3xN np array of points

            # Find range of time where object is likely within iiwa's work evelope
            self.obj_reachable_start_t = 0.2  # random guess
            self.obj_reachable_end_t = 0.5  # random guess
            search_times = np.linspace(0.5, 1, 20)  # assuming first half of trajectory is definitely outside of iiwa's work envelope
            iiwa1_pos = self.iiwa1_pose.translation()
            iiwa2_pos = self.iiwa2_pose.translation()
            # Forward search to find the first time that the object is in iiwa's work envelope
            for t in search_times:
                obj_pose = self.obj_traj.value(t)
                obj_pos = obj_pose.translation()  # (3,) np array containing x,y,z
                obj_dist_from_iiwa1_squared = (obj_pos[0] - iiwa1_pos[0])**2 + (obj_pos[1] - iiwa1_pos[1])**2
                obj_dist_from_iiwa2_squared = (obj_pos[0] - iiwa2_pos[0])**2 + (obj_pos[1] - iiwa2_pos[1])**2
                # Object is between 420-750mm from iiwa's center in XY plane
                if obj_dist_from_iiwa1_squared > 0.42**2 and obj_dist_from_iiwa1_squared < 0.75**2 and obj_dist_from_iiwa2_squared > 0.42**2 and obj_dist_from_iiwa2_squared < 0.75**2:
                    self.obj_reachable_start_t = t
                    break
            # Backward search to find last time
            for t in search_times[::-1]:
                obj_pose = self.obj_traj.value(t)
                obj_pos = obj_pose.translation()  # (3,) np array containing x,y,z
                obj_dist_from_iiwa1_squared = (obj_pos[0] - iiwa1_pos[0])**2 + (obj_pos[1] - iiwa1_pos[1])**2
                obj_dist_from_iiwa2_squared = (obj_pos[0] - iiwa2_pos[0])**2 + (obj_pos[1] - iiwa2_pos[1])**2
                # Object is between 420-750mm from iiwa's center in XY plane
                if obj_dist_from_iiwa1_squared > 0.42**2 and obj_dist_from_iiwa1_squared < 0.75**2 and obj_dist_from_iiwa2_squared > 0.42**2 and obj_dist_from_iiwa2_squared < 0.75**2:
                    self.obj_reachable_end_t = t
                    break

            # For now, all grasps will happen at 0.475 of when obj is in iiwa's work envelope
            obj_catch_t = np.random.uniform(self.obj_reachable_start_t, self.obj_reachable_end_t) #0.7
            print(f'obj_reachable_start_t:{self.obj_reachable_start_t}')
            print(f'obj_reachable_end_t:{self.obj_reachable_end_t}')
            print(f'obj_catch_t:{obj_catch_t}')

            self.obj_pose_at_catch = self.obj_traj.value(obj_catch_t)
            print(f'obj_pose{self.obj_pose_at_catch}')
            obj_vel_at_catch = self.obj_traj.EvalDerivative(t)[:3]
            print(f'velocity:{self.obj_traj.EvalDerivative(t)}')
            end_point = self.obj_pose_at_catch.translation()[:3] + 0.1 * obj_vel_at_catch
            vertices = np.hstack([self.obj_pose_at_catch.translation().reshape(3, 1), end_point.reshape(3, 1)])
            self.meshcat.SetLine("velocity_vector", vertices, 50.0, Rgba(r=1.0, g=0.0, b=0.0, a=1.0) )

            start = time.time()
            grasp_candidates_gripper1, grasp_candidates_gripper2 = self.compute_candidate_grasps(self.obj_pc, obj_pc_centroid, obj_catch_t)
            print(f"-----------grasp sampling runtime: {time.time() - start}")

            # Visualize point cloud
            obj_pc_for_visualization = PointCloud(self.obj_pc)
            if (self.visualize):
                obj_pc_for_visualization.mutable_xyzs()[:] = self.random_transform @ obj_pc_for_visualization.xyzs()
                self.meshcat.SetObject("cloud", obj_pc_for_visualization)

            """
            Iterate through all grasps and select the best based on the heuristics in compute_grasp_cost
            """
            min_cost_1 = float('inf')
            min_cost_grasp_1 = None  # RigidTransform, in object frame
            min_cost_2 = float('inf')
            min_cost_grasp_2 = None
            # for grasp, grasp_cost in grasp_candidates_gripper1.items():

            #     if grasp_cost < min_cost_1:
            #         min_cost_1 = grasp_cost
            #         min_cost_grasp_1 = grasp

            #     # draw all grasp candidates
            #     if (self.visualize):
            #         self.draw_grasp_candidate(grasp, prefix="gripper_1 " + str(time.time()))
            
            # for grasp, grasp_cost in grasp_candidates_gripper2.items():

            #     if grasp_cost < min_cost_2:
            #         min_cost_2 = grasp_cost
            #         min_cost_grasp_2 = grasp

            #     # draw all grasp candidates
            #     if (self.visualize):
            #         self.draw_grasp_candidate(grasp, prefix="gripper_2 " + str(time.time()))
            for (grasp1, grasp_cost1), (grasp2, grasp_cost2) in zip(grasp_candidates_gripper1.items(), grasp_candidates_gripper2.items()):
                # Update the minimum cost and associated grasp for gripper 1
                if grasp_cost1 < min_cost_1:
                    min_cost_1 = grasp_cost1
                    min_cost_grasp_1 = grasp1

                # Update the minimum cost and associated grasp for gripper 2
                if grasp_cost2 < min_cost_2:
                    min_cost_2 = grasp_cost2
                    min_cost_grasp_2 = grasp2
                # print(f'grasp1:{grasp1}, grasp2:{grasp2}')
                # Draw all grasp candidates for both grippers
                if self.visualize:
                    self.draw_grasp_candidate(grasp1, grasp2, prefix=f"grippers {time.time()}",random_transform=False)


            # Convert min_cost_grasp to world frame
            # self.meshcat.SetObject(f"Predobj2", Cylinder(0.02159, 0.9144), Rgba(0,0,1, 1))
            # self.meshcat.SetTransform(f"Predobj2", self.obj_pose_at_catch)
            # self.obj_pose_at_catch = RigidTransform(RollPitchYaw(self.obj_pose_at_catch.rotation()).ToRotationMatrix() @ RollPitchYaw(np.radians([-180,0,0])).ToRotationMatrix(),self.obj_pose_at_catch.translation())
            min_cost_grasp_1_W = self.obj_pose_at_catch @ min_cost_grasp_1
            min_cost_grasp_2_W = self.obj_pose_at_catch @ min_cost_grasp_2
            # draw best grasp gripper position in world
            if (self.visualize):
            #     print(self.obj_pose_at_catch)
                self.draw_grasp_candidate(min_cost_grasp_1_W, min_cost_grasp_2_W, prefix="grippers_best", random_transform=False)
                self.draw_grasp_candidate(min_cost_grasp_1, min_cost_grasp_2, prefix="grippers_best1", random_transform=False)
                # self.draw_grasp_candidate(min_cost_grasp_2_W, prefix="gripper_best_2", random_transform=False)

            output.set_value({'gripper1': (min_cost_grasp_1_W, obj_catch_t), 'gripper2': (min_cost_grasp_2_W, obj_catch_t)})

            # Update class attributes so that next time grasp is not re-selected
            self.selected_grasp1_obj_frame = min_cost_grasp_1
            self.selected_grasp2_obj_frame = min_cost_grasp_2
            self.obj_catch_t = obj_catch_t

        else:
            self.obj_traj = self.get_input_port(1).Eval(context)

            # Allow the estimated catch pose to vary in translation, not rotation, since rotation has so much noise that traj opt will fail
            estimated_obj_catch_pose = RigidTransform(self.obj_pose_at_catch.rotation(), self.obj_traj.value(self.obj_catch_t).translation())
            self.meshcat.SetObject(f"Predobj2", Cylinder(0.02159, 0.9144), Rgba(0,0,1, 1))
            self.meshcat.SetTransform(f"Predobj2", self.obj_traj.value(self.obj_catch_t))
            # Shift selected grasp slightly if object's predicted location at catch time has changed
            selected_grasp1_world_frame = estimated_obj_catch_pose @ self.selected_grasp1_obj_frame
            selected_grasp2_world_frame = estimated_obj_catch_pose @ self.selected_grasp2_obj_frame
            self.draw_grasp_candidate(selected_grasp1_world_frame, selected_grasp2_world_frame, prefix="grippers_best", random_transform=False)
            output.set_value({'gripper1': (selected_grasp1_world_frame, self.obj_catch_t), 'gripper2': (selected_grasp2_world_frame, self.obj_catch_t)})