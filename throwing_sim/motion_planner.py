import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    AbstractValue,
    DiagramBuilder,
    BsplineTrajectory,
    CompositeTrajectory,
    PiecewisePolynomial,
    PathParameterizedTrajectory,
    KinematicTrajectoryOptimization,
    Parser,
    PositionConstraint,
    OrientationConstraint,
    SpatialVelocityConstraint,
    RigidTransform,
    Solve,
    RotationMatrix,
    JacobianWrtVariable,
    RollPitchYaw
)

from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.utils import ConfigureParser
from pydrake.multibody import inverse_kinematics
from pydrake.solvers import SnoptSolver, IpoptSolver

from utils import ObjectTrajectory, calculate_obj_distance_to_gripper


class MotionPlanner(LeafSystem):
    """
    Perform Constrained Optimization to find optimal trajectory for iiwa to move
    to the grasping position.
    """

    def __init__(self, original_plant, meshcat):
        LeafSystem.__init__(self)
        grasp = AbstractValue.Make({'gripper1': (RigidTransform(), 0), 'gripper2': (RigidTransform(), 0)}) # right:gripper1, left:gripper2
        self.DeclareAbstractInputPort("grasp_selection", grasp)                 #0 input_port

        # used to figure out current gripper pose
        body_poses_1 = AbstractValue.Make([RigidTransform()])
        self.DeclareAbstractInputPort("iiwa_current_pose_1", body_poses_1)      #1
        body_poses_2 = AbstractValue.Make([RigidTransform()])
        self.DeclareAbstractInputPort("iiwa_current_pose_2", body_poses_2)      #2

        obj_traj = AbstractValue.Make(ObjectTrajectory())
        self.DeclareAbstractInputPort("object_trajectory", obj_traj)            #3

        iiwa_state_1 = self.DeclareVectorInputPort(name="iiwa_state_1", size=14)  #4      # 7 pos, 7 vel 
        iiwa_state_2 = self.DeclareVectorInputPort(name="iiwa_state_2", size=14)  #5      # 7 pos, 7 vel


        self._traj_index_1 = self.DeclareAbstractState(
            AbstractValue.Make(CompositeTrajectory([PiecewisePolynomial.FirstOrderHold(
                                                        [0, 1],
                                                        np.array([[0, 0]])
                                                    )]))
        )
        self._traj_index_2 = self.DeclareAbstractState(
            AbstractValue.Make(CompositeTrajectory([PiecewisePolynomial.FirstOrderHold(
                                                        [0, 1],
                                                        np.array([[0, 0]])
                                                    )]))
        )



        self.DeclareVectorOutputPort(
            "iiwa_command_1", 14, self.output_traj_1  # 7 pos, 7 vel, robot1
        )
        self.DeclareVectorOutputPort(
            "iiwa_command_2", 14, self.output_traj_2  # 7 pos, 7 vel, robot2
        )

        self.DeclareVectorOutputPort(
            "iiwa_acceleration_1", 7, self.output_acceleration_1
        )
        self.DeclareVectorOutputPort(
            "iiwa_acceleration_2", 7, self.output_acceleration_2
        )

        self.DeclareVectorOutputPort(
            "wsg_command_1", 1, self.output_wsg_traj_1  # 7 pos, 7 vel
        )
        self.DeclareVectorOutputPort(
            "wsg_command_2", 1, self.output_wsg_traj_2  # 7 pos, 7 vel
        )

        self.original_plant = original_plant
        self.meshcat = meshcat
        self.q_nominal_1 = np.array([-2.02981889, 1.31595121, 2.14975521, -1.74829194, -2.53887918, 0.75388712, -0.94564444]) #np.array([-2.09, 0.46, 0.78, -1.78, -0.2, -1.50, 1.3])  # nominal joint for joint-centering
        self.q_nominal_2 = np.array([1.66573262, 1.47287426, -2.23531341, -1.50961309, 0.49256872, -1.3109021, -0.46245998]) #np.array([2.09,  0.46, -0.78, -1.78,  0.2, -1.5,  1.3])
        self.initial_pose_iiwa1 = RigidTransform(RollPitchYaw(np.radians([0, 0, -90])), [2.3, 0.5, 0.0])
        self.initial_pose_iiwa2 = RigidTransform(RollPitchYaw(np.radians([0, 0, 90])), [2.3, -0.5, 0.0])
        self.q_end = None
        self.previous_compute_result_1 = None  # BpslineTrajectory object
        self.previous_compute_result_2 = None 
        self.visualize = True

        self.desired_wsg_1_state = 1  # default open
        self.desired_wsg_2_state = 1

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.025, 0.0, self.compute_traj)

        self.gripper_instance_1 = self.original_plant.GetModelInstanceByName("wsg")
        self.gripper_instance_2 = self.original_plant.GetModelInstanceByName("wsg_2")

        self.obj_catch_t = np.inf
        self.J_1 = np.zeros((3,7))
        self.J_2 = np.zeros((3,7))
        self.FK = True
        self.FK_2 = True
        self.last_pose_1 = None
        self.last_pose_2 = None

    def setSolverSettings(self, prog):
        prog.SetSolverOption(SnoptSolver().solver_id(), "Feasibility tolerance", 0.001)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Major feasibility tolerance", 0.001)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Minor feasibility tolerance", 0.001)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Major optimality tolerance", 0.001)
        prog.SetSolverOption(SnoptSolver().solver_id(), "Minor optimality tolerance", 0.001)


    # Path Visualization
    def VisualizePath(self, traj, name, iiwa):
        """
        Helper function that takes in trajopt basis and control points of Bspline
        and draws spline in meshcat.
        """
        # Build a new plant to do the forward kinematics to turn this Bspline into 3D coordinates
        builder = DiagramBuilder()
        vis_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        viz_iiwa = Parser(vis_plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf")[0]  # ModelInstance object
        # vis_plant.WeldFrames(vis_plant.world_frame(), vis_plant.GetFrameByName("base"))
        if iiwa == 'iiwa_1':
            initial_pose = self.initial_pose_iiwa1 
        else:
            initial_pose = self.initial_pose_iiwa2
        vis_plant.WeldFrames(vis_plant.world_frame(), vis_plant.GetFrameByName("base", viz_iiwa), initial_pose)
        
        vis_plant.Finalize()
        vis_plant_context = vis_plant.CreateDefaultContext()

        traj_start_time = traj.start_time()
        traj_end_time = traj.end_time()

        # Build matrix of 3d positions by doing forward kinematics at time steps in the bspline
        NUM_STEPS = 50
        pos_3d_matrix = np.zeros((3,NUM_STEPS))
        ctr = 0
        for vis_t in np.linspace(traj_start_time, traj_end_time, NUM_STEPS):
            iiwa_pos = traj.value(vis_t)
            vis_plant.SetPositions(vis_plant_context, viz_iiwa, iiwa_pos)
            pos_3d = vis_plant.CalcRelativeTransform(vis_plant_context, vis_plant.world_frame(), vis_plant.GetFrameByName("iiwa_link_7")).translation()
            pos_3d_matrix[:,ctr] = pos_3d
            ctr += 1

        # Draw line
        if self.visualize:
            self.meshcat.SetLine(name, pos_3d_matrix)


    def add_constraints(self,
                        plant,
                        plant_context,
                        plant_autodiff,
                        trajopt,
                        world_frame,
                        gripper_frame,
                        X_WStart,
                        X_WGoal,
                        obj_traj,
                        obj_catch_t,
                        current_gripper_vel,
                        duration_target,
                        acceptable_dur_err=0.01,
                        acceptable_pos_err=0.02,
                        theta_bound = 0.1,
                        acceptable_vel_err=0.4):

        # trajopt.AddPathLengthCost(1.0)

        trajopt.AddPositionBounds(
            plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
        )
        trajopt.AddVelocityBounds(
            plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
        )

        obj_vel_at_catch = obj_traj.EvalDerivative(obj_catch_t)[:3]
        # if object is up to 0.1m radius, make gripper arrive at catching pose that distance early
        # gripper_early_time = 0.08 / np.linalg.norm(obj_vel_at_catch)
        # print(f"gripper_early_time: {gripper_early_time}")
        # duration_target -= gripper_early_time

        pre_catch_time_offset = 0 #0.6
        catch_time_normalized = duration_target / (duration_target + pre_catch_time_offset)
        duration_target += pre_catch_time_offset

        pre_gripper_time = duration_target - pre_catch_time_offset - 0.1
        pre_gripper_normalized_time = pre_gripper_time / duration_target
        X_WO = obj_traj.value(obj_catch_t)
        X_OG_W = X_WO.inverse() @ X_WGoal
        X_WPreGoal = obj_traj.value(obj_catch_t - 0.12) @ X_OG_W

        obj_vel_at_pre_catch = obj_traj.EvalDerivative(obj_catch_t - 0.12)[:3]

        # print(f"duration_target: {duration_target}")
        trajopt.AddDurationConstraint(duration_target-acceptable_dur_err, duration_target+acceptable_dur_err)

        link_7_to_gripper_transform = RotationMatrix.MakeZRotation(np.pi / 2) @ RotationMatrix.MakeXRotation(np.pi / 2)

        # To avoid segfaulting on the constraints being deallocated
        global start_pos_constraint, start_orientation_constraint,\
               pre_goal_pos_constraint, pre_goal_orientation_constraint, pre_goal_vel_constraint,\
               goal_pos_constraint, goal_orientation_constraint, \
               start_vel_constraint, final_vel_constraint

        # start constraint
        start_pos_constraint = PositionConstraint(
            plant,
            world_frame,
            X_WStart.translation() - acceptable_pos_err,  # lower limit
            X_WStart.translation() + acceptable_pos_err,  # upper limit
            gripper_frame,
            [0, 0, 0.1],
            plant_context,
        )
        start_orientation_constraint = OrientationConstraint(
            plant,
            world_frame,
            X_WStart.rotation(),  # orientation of X_WStart in world frame ...
            gripper_frame,
            link_7_to_gripper_transform,  # ... must equal origin in gripper frame
            theta_bound,
            plant_context
        )
        trajopt.AddPathPositionConstraint(start_pos_constraint, 0)
        trajopt.AddPathPositionConstraint(start_orientation_constraint, 0)

        # Pre goal constraint
        # pre_goal_pos_constraint = PositionConstraint(
        #     plant,
        #     world_frame,
        #     X_WPreGoal.translation() - max(acceptable_pos_err * 25, 0.2),  # lower limit
        #     X_WPreGoal.translation() + max(acceptable_pos_err * 25, 0.2),  # upper limit
        #     gripper_frame,
        #     [0, 0, 0.1],
        #     plant_context
        # )
        # pre_goal_orientation_constraint = OrientationConstraint(
        #     plant,
        #     world_frame,
        #     X_WPreGoal.rotation(),  # orientation of X_WGoal in world frame ...
        #     gripper_frame,
        #     link_7_to_gripper_transform,  # ... must equal origin in gripper frame
        #     theta_bound,
        #     plant_context
        # )
        # trajopt.AddPathPositionConstraint(pre_goal_pos_constraint, pre_gripper_normalized_time)
        # trajopt.AddPathPositionConstraint(pre_goal_orientation_constraint, pre_gripper_normalized_time)

        # goal constraint
        goal_pos_constraint = PositionConstraint(
            plant,
            world_frame,
            X_WGoal.translation() - acceptable_pos_err,  # lower limit
            X_WGoal.translation() + acceptable_pos_err,  # upper limit
            gripper_frame,
            [0, 0, 0.1],
            plant_context,
        )
        goal_orientation_constraint = OrientationConstraint(
            plant,
            world_frame,
            X_WGoal.rotation(),  # orientation of X_WGoal in world frame ...
            gripper_frame,
            link_7_to_gripper_transform,  # ... must equal origin in gripper frame
            theta_bound,
            plant_context
        )
        # print(f"catch_time_normalized: {catch_time_normalized}")
        trajopt.AddPathPositionConstraint(goal_pos_constraint, pre_gripper_normalized_time)
        trajopt.AddPathPositionConstraint(goal_orientation_constraint, pre_gripper_normalized_time)

        goal_pos_constraint = PositionConstraint(
            plant,
            world_frame,
            X_WGoal.translation() - acceptable_pos_err,  # lower limit
            X_WGoal.translation() + acceptable_pos_err,  # upper limit
            gripper_frame,
            [0, 0, 0.1],
            plant_context,
        )
        goal_orientation_constraint = OrientationConstraint(
            plant,
            world_frame,
            X_WGoal.rotation(),  # orientation of X_WGoal in world frame ...
            gripper_frame,
            link_7_to_gripper_transform,  # ... must equal origin in gripper frame
            theta_bound,
            plant_context
        )
        # print(f"catch_time_normalized: {catch_time_normalized}")
        trajopt.AddPathPositionConstraint(goal_pos_constraint, 1)
        trajopt.AddPathPositionConstraint(goal_orientation_constraint, 1)

        # Start with velocity equal to iiwa's current velocity
        # Current limitation: SpatialVelocityConstraint only takes into account translational velocity; not rotational
        # start_vel_constraint = SpatialVelocityConstraint(
        #     plant_autodiff,
        #     plant_autodiff.world_frame(),
        #     current_gripper_vel - acceptable_vel_err,  # upper limit
        #     current_gripper_vel + acceptable_vel_err,  # lower limit
        #     plant_autodiff.GetFrameByName("iiwa_link_7"),
        #     np.array([0, 0, 0.1]).reshape(-1,1),
        #     plant_autodiff.CreateDefaultContext(),
        # )

        # # end with same directional velocity as object at catch
        # catch_vel = obj_vel_at_catch * 0.3 # (3,1) np array
        # final_vel_constraint = SpatialVelocityConstraint(
        #     plant_autodiff,
        #     plant_autodiff.world_frame(),
        #     catch_vel - acceptable_vel_err,  # upper limit
        #     catch_vel + acceptable_vel_err,  # lower limit
        #     plant_autodiff.GetFrameByName("iiwa_link_7"),
        #     np.array([0, 0, 0.1]).reshape(-1,1),
        #     plant_autodiff.CreateDefaultContext(),
        # )

        # trajopt.AddVelocityConstraintAtNormalizedTime(start_vel_constraint, 0)
        # trajopt.AddVelocityConstraintAtNormalizedTime(final_vel_constraint, catch_time_normalized)


    def compute_traj(self, context, state):
        print("motion_planner update event")
        # if self.previous_compute_result != None:
        #     return

        obj_traj = self.get_input_port(3).Eval(context)
        if (obj_traj == ObjectTrajectory()):  # default output of TrajectoryPredictor system; means that it hasn't seen the object yet
            # print("received default obj traj (in compute_traj). returning from compute_traj.")
            return

        # Get current gripper pose from input port
        body_poses_1 = self.get_input_port(1).Eval(context)  # "iiwa_current_pose_1" input port
        gripper_1_body_idx = self.original_plant.GetBodyByName("body_1", self.gripper_instance_1).index()  # BodyIndex object
        current_gripper_pose_1 = body_poses_1[gripper_1_body_idx]  # RigidTransform object
        body_poses_2 = self.get_input_port(2).Eval(context)  # "iiwa_current_pose_2" input port
        gripper_2_body_idx = self.original_plant.GetBodyByName("body_2", self.gripper_instance_2).index()  # BodyIndex object
        current_gripper_pose_2 = body_poses_2[gripper_2_body_idx]  # RigidTransform object
        if self.visualize:
            AddMeshcatTriad(self.meshcat, "goal1", X_PT=current_gripper_pose_1, opacity=0.5)
            self.meshcat.SetTransform("goal1", current_gripper_pose_1)
            AddMeshcatTriad(self.meshcat, "goal2", X_PT=current_gripper_pose_2, opacity=0.5)
            self.meshcat.SetTransform("goal2", current_gripper_pose_2)
        # Get current iiwa positions and velocities
        iiwa_state_1 = self.get_input_port(4).Eval(context) # "iiwa_state_1"
        q_current_1 = iiwa_state_1[:7]
        iiwa_vels_1 = iiwa_state_1[7:]
        iiwa_state_2 = self.get_input_port(5).Eval(context) # "iiwa_state_2"
        q_current_2 = iiwa_state_2[:7]
        iiwa_vels_2 = iiwa_state_2[7:]

        # Build a new plant to do calculate the velocity Jacobian
        builder = DiagramBuilder()
        j_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        j_iiwa_1 = Parser(j_plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf")[0]  # ModelInstance object
        j_plant.WeldFrames(j_plant.world_frame(), j_plant.GetFrameByName("base", j_iiwa_1))
        j_plant.Finalize()
        j_plant_context = j_plant.CreateDefaultContext()
        j_plant.SetPositions(j_plant_context, j_iiwa_1, q_current_1)
        # Build Jacobian to solve for translational velocity from joint velocities
        self.J_1 = j_plant.CalcJacobianTranslationalVelocity(j_plant_context,
                                                      JacobianWrtVariable.kQDot,
                                                      j_plant.GetFrameByName("iiwa_link_7", j_iiwa_1),
                                                      [0, 0, 0.1],  # offset from iiwa_link_7_ to where gripper would be
                                                      j_plant.world_frame(),  #
                                                      j_plant.world_frame()  # frame that translational velocity should be expressed in
                                                      )
        current_gripper_1_vel = np.dot(self.J_1, iiwa_vels_1)

        builder = DiagramBuilder()
        j_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        j_iiwa_2 = Parser(j_plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision_2.urdf")[0]  # ModelInstance object
        j_plant.WeldFrames(j_plant.world_frame(), j_plant.GetFrameByName("base", j_iiwa_2))
        j_plant.Finalize()
        j_plant_context = j_plant.CreateDefaultContext()
        j_plant.SetPositions(j_plant_context, j_iiwa_2, q_current_2)
        self.J_2 = j_plant.CalcJacobianTranslationalVelocity(j_plant_context,
                                                      JacobianWrtVariable.kQDot,
                                                      j_plant.GetFrameByName("iiwa_link_7", j_iiwa_2),
                                                      [0, 0, 0.1],  # offset from iiwa_link_7_ to where gripper would be
                                                      j_plant.world_frame(),  #
                                                      j_plant.world_frame()  # frame that translational velocity should be expressed in
                                                      )
        current_gripper_2_vel = np.dot(self.J_2, iiwa_vels_2)
        

        # Get selected grasp pose from input port
        grasp = self.get_input_port(0).Eval(context)
        X_WG1, obj_catch_t = grasp['gripper1']
        X_WG2, obj_catch_t = grasp['gripper2']
        self.obj_catch_t = obj_catch_t if obj_catch_t != 0 else np.inf
        if (X_WG1.IsExactlyEqualTo(RigidTransform()) or X_WG2.IsExactlyEqualTo(RigidTransform())):
            print("received default catch pose. returning from compute_traj.")
            return
        # print(f"obj_catch_t: {obj_catch_t}")
        # print(obj_traj)
        # print(obj_traj.value(self.obj_catch_t))
        # print(f'X_WG1:{X_WG1}')
        # print(f'X_WG2:{X_WG2}')
        # If it's getting close to catch time, stop updating trajectory
        if obj_catch_t - context.get_time() < 0.2:
            return

        # Setup a new MBP with just the iiwa which the KinematicTrajectoryOptimization will use
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        iiwa_1 = Parser(plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf")[0]  # ModelInstance object
        world_frame = plant.world_frame()
        base_frame_1 = plant.GetFrameByName("base", iiwa_1)
        gripper_frame_1 = plant.GetFrameByName("iiwa_link_7", iiwa_1)
        # initial_pose = RigidTransform(RollPitchYaw(np.radians([0, 0, -90])), [2.3, 0.5, 0.0])
        plant.WeldFrames(world_frame, base_frame_1, self.initial_pose_iiwa1)  # Weld iiwa to world

        plant.Finalize()
        plant_context = plant.CreateDefaultContext()
        plant.SetPositions(plant_context, iiwa_1, self.q_nominal_1)

        # Create auto-differentiable version the plant in order to set velocity constraints
        plant_autodiff = plant.ToAutoDiffXd()

        X_W1Start = current_gripper_pose_1
        X_W1Goal = X_WG1
        
        # print(f"X_WStart: {X_WStart}")
        # print(f"X_WGoal: {X_WGoal}")
        # if self.visualize:
        #     # AddMeshcatTriad(self.meshcat, "start", X_PT=X_WStart, opacity=0.5)
        #     # self.meshcat.SetTransform("start", X_WStart)
        #     AddMeshcatTriad(self.meshcat, "goal", X_PT=X_W1Goal, opacity=0.5)
        #     self.meshcat.SetTransform("goal", X_W1Goal)
        

        obj_vel_at_catch = obj_traj.EvalDerivative(obj_catch_t)[:3]  # (3,1) np array
        # print(f"current_gripper_1_vel: {current_gripper_1_vel}", f"current_gripper_2_vel: {current_gripper_2_vel}")
        # print(f"obj_vel_at_catch: {obj_vel_at_catch}")

        num_q_1 = plant.num_positions(iiwa_1)  # =7 (all of iiwa's joints)
        

        # If this is the very first traj opt (so we don't yet have a very good initial guess), do an interative optimization
        MAX_ITERATIONS = 12
        ###################################                     
        ###     trajectory for iiwa1    ### 
        ###################################  
        if self.previous_compute_result_1 is None:
            num_iter = 0
            cur_acceptable_duration_err=0.05
            cur_acceptable_pos_err=0.1
            cur_theta_bound=0.8
            cur_acceptable_vel_err=2.0
            final_traj_1 = None
            while(num_iter < MAX_ITERATIONS):
                trajopt = KinematicTrajectoryOptimization(num_q_1, 8)  # 8 control points in Bspline
                prog = trajopt.get_mutable_prog()
                self.setSolverSettings(prog)

                if num_iter == 0:
                    print("using ik for initial guess")
                    # First solve the IK problem for X_WGoal. Then lin interp from start pos to goal pos,
                    # use these points as control point initial guesses for the optimization.
                    ik = inverse_kinematics.InverseKinematics(plant)
                    q_variables = ik.q()  # Get variables for MathematicalProgram
                    ik_prog = ik.prog()
                    ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), self.q_nominal_1, q_variables)
                    ik.AddPositionConstraint(
                        frameA=world_frame,
                        frameB=gripper_frame_1,
                        p_BQ=[0, 0, 0.1],
                        p_AQ_lower=X_W1Goal.translation(),
                        p_AQ_upper=X_W1Goal.translation(),
                    )
                    ik.AddOrientationConstraint(
                        frameAbar=world_frame,
                        R_AbarA=X_W1Goal.rotation(),
                        frameBbar=gripper_frame_1,
                        R_BbarB=RotationMatrix(),
                        theta_bound=0.05,
                    )
                    ik_prog.SetInitialGuess(q_variables, self.q_nominal_1)
                    ik_result = Solve(ik_prog)
                    if not ik_result.is_success():
                        # print('X_W1Goal', X_W1Goal)
                        print("ERROR: ik_result_1 solve failed: " + str(ik_result.get_solver_id().name()))
                        print(ik_result.GetInfeasibleConstraintNames(ik_prog))
                    else:
                        print("ik_result_1 solve succeeded.")

                    q_end = ik_result.GetSolution(q_variables)
                    # Guess 8 control points in 7D for Bspline
                    q_guess = np.linspace(q_current_1, q_end, 8).T  # (7,8) np array
                    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
                    trajopt.SetInitialGuess(path_guess)
                else:
                    print("using previous iter as initial guess")
                    trajopt.SetInitialGuess(final_traj_1)

                self.add_constraints(plant,
                                     plant_context,
                                     plant_autodiff,
                                     trajopt,
                                     world_frame,
                                     gripper_frame_1,
                                     X_W1Start,
                                     X_W1Goal,
                                     obj_traj,
                                     obj_catch_t,
                                     current_gripper_1_vel,
                                     duration_target=obj_catch_t-context.get_time(),
                                     acceptable_dur_err=cur_acceptable_duration_err,
                                     acceptable_pos_err=cur_acceptable_pos_err,
                                     theta_bound=cur_theta_bound,
                                     acceptable_vel_err=cur_acceptable_vel_err)
                
                # First solve with looser constraints
                solver = SnoptSolver()
                result = solver.Solve(prog)
                if not result.is_success():
                    print(f"ERROR: num_iter={num_iter} Trajectory optimization 1 failed: {result.get_solver_id().name()}")
                    print(result.GetInfeasibleConstraintNames(prog))
                    if final_traj_1 is None:  # ensure final_traj is not None
                        final_traj_1 = trajopt.ReconstructTrajectory(result)
                    # raise RuntimeError()
                    break
                else:
                    print(f"num_iter={num_iter} Solve succeeded.")

                final_traj_1 = trajopt.ReconstructTrajectory(result)  # BSplineTrajectory

                self.VisualizePath(final_traj_1, f"traj_1 iter={num_iter}", 'iiwa_1')

                # Make constraints more strict next iteration
                cur_acceptable_duration_err *= 0.875
                cur_acceptable_pos_err *= 0.875
                cur_theta_bound *= 0.875
                cur_acceptable_vel_err *= 0.875

                num_iter += 1

        # If this is not the first cycle (so we have a good initial guess already), then just go straight to an optimization w/strict constraints
        else:
            # Undraw previous trajctories that aren't actually being followed
            for i in range(MAX_ITERATIONS):
                try:
                    if self.visualize:
                        self.meshcat.Delete(f"traj_1 iter={i}")
                except:
                    pass

            print("using previous cycle's executed trajectory as initial guess")
            trajopt = KinematicTrajectoryOptimization(num_q_1, 8)  # 8 control points in Bspline
            prog = trajopt.get_mutable_prog()
            self.setSolverSettings(prog)
            self.add_constraints(plant,
                                 plant_context,
                                 plant_autodiff,
                                 trajopt,
                                 world_frame,
                                 gripper_frame_1,
                                 X_W1Start,
                                 X_W1Goal,
                                 obj_traj,
                                 obj_catch_t,
                                 current_gripper_1_vel,
                                 duration_target=obj_catch_t-context.get_time())

            trajopt.SetInitialGuess(self.previous_compute_result_1)

            solver = SnoptSolver()
            result = solver.Solve(prog)
            if not result.is_success():
                print("ERROR: Tight Trajectory optimization 1 failed: " + str(result.get_solver_id().name()))
                print(result.GetInfeasibleConstraintNames(prog))
            else:
                print("Tight solve 1 succeeded.")

            final_traj_1 = trajopt.ReconstructTrajectory(result)  # BSplineTrajectory

        # Shift trajectory in time so that it starts at the current time
        time_shift = context.get_time()  # Time shift value in seconds
        time_scaling_traj = PiecewisePolynomial.FirstOrderHold(
            [time_shift, time_shift+final_traj_1.end_time()],  # Assuming two segments: initial and final times
            np.array([[0, final_traj_1.end_time()-final_traj_1.start_time()]])  # Shifts start and end times by time_shift
        )
        time_shifted_final_traj_1 = PathParameterizedTrajectory(
            final_traj_1, time_scaling_traj
        )
        # print(f"time_shifted_final_traj.start_time(): {time_shifted_final_traj.start_time()}")
        # print(f"time_shifted_final_traj.end_time(): {time_shifted_final_traj.end_time()}")

        self.VisualizePath(time_shifted_final_traj_1, "final traj", 'iiwa_1')

        state.get_mutable_abstract_state(int(self._traj_index_1)).set_value(time_shifted_final_traj_1)

        self.previous_compute_result_1 = final_traj_1  # save the solved trajectory to use as initial guess next iteration



        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        iiwa_2 = Parser(plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision_2.urdf")[0]  # ModelInstance object
        world_frame = plant.world_frame()
        base_frame_2 = plant.GetFrameByName("base", iiwa_2)
        gripper_frame_2 = plant.GetFrameByName("iiwa_link_7", iiwa_2)
        # initial_pose_2 = RigidTransform(RollPitchYaw(np.radians([0, 0, 90])), [2.3, -0.5, 0.0])
        plant.WeldFrames(world_frame, base_frame_2, self.initial_pose_iiwa2)
        
        plant.Finalize()
        plant_context = plant.CreateDefaultContext()
        plant.SetPositions(plant_context, iiwa_2, self.q_nominal_2)

        # Create auto-differentiable version the plant in order to set velocity constraints
        plant_autodiff = plant.ToAutoDiffXd()

        X_W2Start = current_gripper_pose_2
        X_W2Goal = X_WG2

        # if self.visualize:
        #     AddMeshcatTriad(self.meshcat, "goal_2", X_PT=X_W2Goal, opacity=0.5)
        #     self.meshcat.SetTransform("goal_2", X_W2Goal)

        obj_vel_at_catch = obj_traj.EvalDerivative(obj_catch_t)[:3]  # (3,1) np array
        num_q_2 = plant.num_positions(iiwa_2)
        ###################################                     
        ###     trajectory for iiwa2    ### 
        ################################### 
        if self.previous_compute_result_2 is None:
            num_iter = 0
            cur_acceptable_duration_err=0.05
            cur_acceptable_pos_err=0.1
            cur_theta_bound=0.8
            cur_acceptable_vel_err=2.0
            final_traj_2 = None
            while(num_iter < MAX_ITERATIONS):
                trajopt = KinematicTrajectoryOptimization(num_q_2, 8)  # 8 control points in Bspline
                prog = trajopt.get_mutable_prog()
                self.setSolverSettings(prog)

                if num_iter == 0:
                    print("using ik for initial guess 2")
                    # First solve the IK problem for X_WGoal. Then lin interp from start pos to goal pos,
                    # use these points as control point initial guesses for the optimization.
                    ik = inverse_kinematics.InverseKinematics(plant)
                    q_variables = ik.q()  # Get variables for MathematicalProgram
                    ik_prog = ik.prog()
                    ik_prog.AddQuadraticErrorCost(np.identity(len(q_variables)), self.q_nominal_2, q_variables)
                    ik.AddPositionConstraint(
                        frameA=world_frame,
                        frameB=gripper_frame_2,
                        p_BQ=[0, 0, 0.1],
                        p_AQ_lower=X_W2Goal.translation(),
                        p_AQ_upper=X_W2Goal.translation(),
                    )
                    ik.AddOrientationConstraint(
                        frameAbar=world_frame,
                        R_AbarA=X_W2Goal.rotation(),
                        frameBbar=gripper_frame_2,
                        R_BbarB=RotationMatrix(),
                        theta_bound=0.05,
                    )
                    ik_prog.SetInitialGuess(q_variables, self.q_nominal_2)
                    ik_result = Solve(ik_prog)
                    if not ik_result.is_success():
                        # print('X_W2Goal', X_W2Goal)
                        print("ERROR: ik_result_2 solve failed: " + str(ik_result.get_solver_id().name()))
                        print(ik_result.GetInfeasibleConstraintNames(ik_prog))
                    else:
                        print("ik_result_2 solve succeeded.")

                    q_end = ik_result.GetSolution(q_variables)
                    # Guess 8 control points in 7D for Bspline
                    q_guess = np.linspace(q_current_2, q_end, 8).T  # (7,8) np array
                    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
                    trajopt.SetInitialGuess(path_guess)
                else:
                    print("using previous iter as initial guess")
                    trajopt.SetInitialGuess(final_traj_2)

                self.add_constraints(plant,
                                     plant_context,
                                     plant_autodiff,
                                     trajopt,
                                     world_frame,
                                     gripper_frame_2,
                                     X_W2Start,
                                     X_W2Goal,
                                     obj_traj,
                                     obj_catch_t,
                                     current_gripper_2_vel,
                                     duration_target=obj_catch_t-context.get_time(),            # might cause problem?
                                     acceptable_dur_err=cur_acceptable_duration_err,
                                     acceptable_pos_err=cur_acceptable_pos_err,
                                     theta_bound=cur_theta_bound,
                                     acceptable_vel_err=cur_acceptable_vel_err)
                
                # First solve with looser constraints
                solver = SnoptSolver()
                result = solver.Solve(prog)
                if not result.is_success():
                    print(f"ERROR: num_iter={num_iter} Trajectory optimization 2 failed: {result.get_solver_id().name()}")
                    print(result.GetInfeasibleConstraintNames(prog))
                    if final_traj_2 is None:  # ensure final_traj is not None
                        final_traj_2 = trajopt.ReconstructTrajectory(result)
                    # raise RuntimeError()
                    break
                else:
                    print(f"num_iter={num_iter} Solve succeeded.")

                final_traj_2 = trajopt.ReconstructTrajectory(result)  # BSplineTrajectory

                self.VisualizePath(final_traj_2, f"traj_2 iter={num_iter}", 'iiwa_2')

                # Make constraints more strict next iteration
                cur_acceptable_duration_err *= 0.875
                cur_acceptable_pos_err *= 0.875
                cur_theta_bound *= 0.875
                cur_acceptable_vel_err *= 0.875

                num_iter += 1

        # If this is not the first cycle (so we have a good initial guess already), then just go straight to an optimization w/strict constraints
        else:
            # Undraw previous trajctories that aren't actually being followed
            for i in range(MAX_ITERATIONS):
                try:
                    if self.visualize:
                        self.meshcat.Delete(f"traj_2 iter={i}")
                except:
                    pass

            print("using previous cycle's executed trajectory as initial guess")
            trajopt = KinematicTrajectoryOptimization(num_q_2, 8)  # 8 control points in Bspline
            prog = trajopt.get_mutable_prog()
            self.setSolverSettings(prog)
            self.add_constraints(plant,
                                 plant_context,
                                 plant_autodiff,
                                 trajopt,
                                 world_frame,
                                 gripper_frame_2,
                                 X_W2Start,
                                 X_W2Goal,
                                 obj_traj,
                                 obj_catch_t,
                                 current_gripper_2_vel,
                                 duration_target=obj_catch_t-context.get_time())

            trajopt.SetInitialGuess(self.previous_compute_result_2)

            solver = SnoptSolver()
            result = solver.Solve(prog)
            if not result.is_success():
                print("ERROR: Tight Trajectory optimization 2 failed: " + str(result.get_solver_id().name()))
                print(result.GetInfeasibleConstraintNames(prog))
            else:
                print("Tight solve 2 succeeded.")

            final_traj_2 = trajopt.ReconstructTrajectory(result)  # BSplineTrajectory

        # Shift trajectory in time so that it starts at the current time
        time_shift = context.get_time()  # Time shift value in seconds
        time_scaling_traj = PiecewisePolynomial.FirstOrderHold(
            [time_shift, time_shift+final_traj_2.end_time()],  # Assuming two segments: initial and final times
            np.array([[0, final_traj_2.end_time()-final_traj_2.start_time()]])  # Shifts start and end times by time_shift
        )
        time_shifted_final_traj_2 = PathParameterizedTrajectory(
            final_traj_2, time_scaling_traj
        )
        # print(f"time_shifted_final_traj.start_time(): {time_shifted_final_traj.start_time()}")
        # print(f"time_shifted_final_traj.end_time(): {time_shifted_final_traj.end_time()}")

        self.VisualizePath(time_shifted_final_traj_2, "final traj_2", 'iiwa_2')

        state.get_mutable_abstract_state(int(self._traj_index_2)).set_value(time_shifted_final_traj_2)

        self.previous_compute_result_2 = final_traj_2  # save the solved trajectory to use as initial guess next iteration
        
        self.obj_vel_at_catch = obj_traj.EvalDerivative(obj_catch_t)[:3]

        #############################
        ###    collision check    ###
        #############################
        ##TODO




    def output_traj_1(self, context, output):
        # Just set value at output port according to context time and trajectory state variable
        traj_q = context.get_mutable_abstract_state(int(self._traj_index_1)).get_value()
        # traj_q.rows() == 1 basically means traj_q is the default;
        # either object trajectory hasn't finished predicting yet, or grasp hasn't been selected yet,
        if (traj_q.rows() == 1):
            # print("planner outputting default iiwa position")
            output.SetFromVector(np.append(
                self.original_plant.GetPositions(self.original_plant.CreateDefaultContext(), self.original_plant.GetModelInstanceByName("iiwa")),
                np.zeros((7,))
            ))
        elif context.get_time() >= self.obj_catch_t + 0.01:
            output.SetFromVector(self.last_pose_1)

        else:
            # if context.get_time() <= traj_q.end_time():
            #     # print("planner outputting iiwa position: " + str(traj_q.value(context.get_time())))
            #     output.SetFromVector(np.append(
            #         traj_q.value(context.get_time()),
            #         traj_q.EvalDerivative(context.get_time())
            #     ))
            # else:  # return the ik result computed at end position
            #     if self.q_end is not None:
            #         output.SetFromVector(np.append(self.q_end, np.zeros(7)))
            # print(f'traj_1{traj_q.end_time()}:{np.dot(self.J_1, traj_q.value(traj_q.end_time()))}')
            output.SetFromVector(np.append(
                traj_q.value(context.get_time()),
                traj_q.EvalDerivative(context.get_time())
            ))
            self.last_pose_1 = np.append(
                traj_q.value(context.get_time()),
                np.zeros((7,)))
            if self.FK == True:
                self.FK = False
                builder = DiagramBuilder()
                j_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
                j_iiwa_2 = Parser(j_plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision_2.urdf")[0]  # ModelInstance object
                gripper_frame = j_plant.GetFrameByName("iiwa_link_7", j_iiwa_2)
                j_plant.WeldFrames(j_plant.world_frame(), j_plant.GetFrameByName("base", j_iiwa_2), self.initial_pose_iiwa1)
                j_plant.Finalize()
                j_plant_context = j_plant.CreateDefaultContext()
                j_plant.SetPositions(j_plant_context, j_iiwa_2, traj_q.value(self.obj_catch_t ))
                gripper_pose = j_plant.CalcRelativeTransform(j_plant_context, frame_A=j_plant.world_frame(), frame_B=gripper_frame)
                # print(f'computed ee pos_1:{gripper_pose}, joint:{traj_q.value(self.obj_catch_t )}')
    

    def output_traj_2(self, context, output):
        # Just set value at output port according to context time and trajectory state variable
        traj_q = context.get_mutable_abstract_state(int(self._traj_index_2)).get_value()
        # traj_q.rows() == 1 basically means traj_q is the default;
        # either object trajectory hasn't finished predicting yet, or grasp hasn't been selected yet,
        if (traj_q.rows() == 1):
            # print("planner outputting default iiwa position")
            output.SetFromVector(np.append(
                self.original_plant.GetPositions(self.original_plant.CreateDefaultContext(), self.original_plant.GetModelInstanceByName("iiwa_2")),
                np.zeros((7,))
            ))
        elif context.get_time() >= self.obj_catch_t+ 0.01:
            output.SetFromVector(self.last_pose_2)
        else:
            # if context.get_time() <= traj_q.end_time():
            #     # print("planner outputting iiwa position: " + str(traj_q.value(context.get_time())))
            #     output.SetFromVector(np.append(
            #         traj_q.value(context.get_time()),
            #         traj_q.EvalDerivative(context.get_time())
            #     ))
            # else:  # return the ik result computed at end position
            #     if self.q_end is not None:
            #         output.SetFromVector(np.append(self.q_end, np.zeros(7)))
            # print(f'traj_2{traj_q.end_time()}:{np.dot(self.J_2, traj_q.value(traj_q.end_time()))}')
            output.SetFromVector(np.append(
                traj_q.value(context.get_time()),
                traj_q.EvalDerivative(context.get_time())
            ))
            self.last_pose_2 = np.append(
                traj_q.value(context.get_time()),
                np.zeros((7,)))
            if self.FK_2 == True:
                self.FK_2 = False
                builder = DiagramBuilder()
                j_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
                j_iiwa_2 = Parser(j_plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision_2.urdf")[0]  # ModelInstance object
                gripper_frame = j_plant.GetFrameByName("iiwa_link_7", j_iiwa_2)
                j_plant.WeldFrames(j_plant.world_frame(), j_plant.GetFrameByName("base", j_iiwa_2), self.initial_pose_iiwa2)
                j_plant.Finalize()
                j_plant_context = j_plant.CreateDefaultContext()
                j_plant.SetPositions(j_plant_context, j_iiwa_2, traj_q.value(self.obj_catch_t))
                gripper_pose = j_plant.CalcRelativeTransform(j_plant_context, frame_A=j_plant.world_frame(), frame_B=gripper_frame)
                # print(f'computed ee pos_2:{gripper_pose}, joint:{traj_q.value(self.obj_catch_t)}')
                


    def output_acceleration_1(self, context, output):
        traj_q = context.get_mutable_abstract_state(int(self._traj_index_1)).get_value()

        if (traj_q.rows() == 1) or context.get_time() >= self.obj_catch_t:
            # print("planner outputting default 0 acceleration")
            output.SetFromVector(np.zeros((7,)))
        else:
            output.SetFromVector(1*(traj_q.EvalDerivative(context.get_time(), 2)))


    def output_acceleration_2(self, context, output):
        traj_q = context.get_mutable_abstract_state(int(self._traj_index_2)).get_value()

        if (traj_q.rows() == 1) or context.get_time() >= self.obj_catch_t:
            # print("planner outputting default 0 acceleration")
            output.SetFromVector(np.zeros((7,)))
        else:
            output.SetFromVector(1*(traj_q.EvalDerivative(context.get_time(), 2)))


    def output_wsg_traj_1(self, context, output):
        # Get current gripper pose from input port
        body_poses = self.get_input_port(1).Eval(context)  # "iiwa_current_pose_1" input port
        # gripper_instance_1 = self.original_plant.GetModelInstanceByName("wsg")
        gripper_body_idx = self.original_plant.GetBodyByName("body_1", self.gripper_instance_1).index()  # BodyIndex object
        current_gripper_pose = body_poses[gripper_body_idx]  # RigidTransform object
        # find body index of obj being thrown
        if self.original_plant.HasBodyNamed("noodle"):
            obj_body_name = "noodle"
        elif self.original_plant.HasBodyNamed("ring"):
            obj_body_name = "ring"
        elif self.original_plant.HasBodyNamed("cuboid"):
            obj_body_name = "cuboid"
        # elif self.original_plant.HasBodyNamed("pill_bottle"):
        #     obj_body_name = "pill_bottle"

        # Get current object pose from input port
        obj_body_idx = self.original_plant.GetBodyByName(obj_body_name).index()  # BodyIndex object
        current_obj_pose = body_poses[obj_body_idx]  # RigidTransform object

        obj_distance_to_grasp, vector_gripper_to_obj = calculate_obj_distance_to_gripper(current_gripper_pose, current_obj_pose)
        # print(f'distance_1:{obj_distance_to_grasp}')
        DISTANCE_TRESHOLD_TO_GRASP = 0.03

        # first comparison measures distance in general (to ensure obj is roughly in range for a catch); 
        # second comparison is more precise, measures obj distance to grasp in gripper frame y-axis
        # if (np.linalg.norm(vector_gripper_to_obj) < 0.25 and obj_distance_to_grasp < DISTANCE_TRESHOLD_TO_GRASP) or self.desired_wsg_1_state == 0:
        # if obj_distance_to_grasp < DISTANCE_TRESHOLD_TO_GRASP or self.desired_wsg_1_state == 0 or context.get_time() >= self.obj_catch_t:
        if self.desired_wsg_1_state == 0 or context.get_time() >= self.obj_catch_t + 0.01:
            output.SetFromVector(np.array([-60]))  # closed
            self.desired_wsg_1_state = 0  # grippers should be closed from now on
        else:
            output.SetFromVector(np.array([1]))  # open


    def output_wsg_traj_2(self, context, output):
        # Get current gripper pose from input port
        body_poses = self.get_input_port(2).Eval(context)  # "iiwa_current_pose_2" input port
        # gripper_instance_2 = self.original_plant.GetModelInstanceByName("wsg_2")
        gripper_body_idx = self.original_plant.GetBodyByName("body_2", self.gripper_instance_2).index()  # BodyIndex object
        current_gripper_pose = body_poses[gripper_body_idx]  # RigidTransform object
        # find body index of obj being thrown
        if self.original_plant.HasBodyNamed("noodle"):
            obj_body_name = "noodle"
        elif self.original_plant.HasBodyNamed("ring"):
            obj_body_name = "ring"
        elif self.original_plant.HasBodyNamed("cuboid"):
            obj_body_name = "cuboid"
        # elif self.original_plant.HasBodyNamed("pill_bottle"):
        #     obj_body_name = "pill_bottle"

        # Get current object pose from input port
        obj_body_idx = self.original_plant.GetBodyByName(obj_body_name).index()  # BodyIndex object
        current_obj_pose = body_poses[obj_body_idx]  # RigidTransform object

        obj_distance_to_grasp, vector_gripper_to_obj = calculate_obj_distance_to_gripper(current_gripper_pose, current_obj_pose)
        # print(f'distance_2:{obj_distance_to_grasp}')
        # print(f'distance_2_vec:{np.linalg.norm(vector_gripper_to_obj)}')
        # print(context.get_time())
        DISTANCE_TRESHOLD_TO_GRASP = 0.03

        # first comparison measures distance in general (to ensure obj is roughly in range for a catch); 
        # second comparison is more precise, measures obj distance to grasp in gripper frame y-axis
        # if (np.linalg.norm(vector_gripper_to_obj) < 0.25 and obj_distance_to_grasp < DISTANCE_TRESHOLD_TO_GRASP) or self.desired_wsg_2_state == 0:
        # if obj_distance_to_grasp < DISTANCE_TRESHOLD_TO_GRASP or self.desired_wsg_2_state == 0 or context.get_time() >= self.obj_catch_t:
        if self.desired_wsg_2_state == 0 or context.get_time() >= self.obj_catch_t + 0.01:
            output.SetFromVector(np.array([-60]))  # closed
            self.desired_wsg_2_state = 0  # grippers should be closed from now on
        else:
            output.SetFromVector(np.array([1]))  # open