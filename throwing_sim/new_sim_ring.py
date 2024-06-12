# This file is licensed under the MIT-0 License.
# See LICENSE-MIT-0.txt in the current directory.

"""
This program serves as an example of a simulator for hardware, i.e., a
simulator for robots that one might have in their lab. There is no controller
built-in to this program -- it merely sends status and sensor messages, and
listens for command messages.

It is intended to operate in the "no ground truth" regime, i.e, the only LCM
messages it knows about are the ones used by the actual hardware. The one
messaging difference from real life is that we emit visualization messages (for
Meldis, or the legacy ``drake-visualizer`` application of days past) so that
you can watch a simulation on your screen while some (separate) controller
operates the robot, without extra hassle.

Drake maintainers should keep this file in sync with both hardware_sim.cc and
scenario.h.
"""

import argparse
import dataclasses as dc
import math
import typing
import matplotlib.pyplot as plt
import numpy as np
import gc
from manipulation.scenarios import AddIiwa
from pydrake.systems.primitives import Demultiplexer
from pydrake.all import (
    RigidTransform, 
    RollPitchYaw, 
    RgbdSensor, 
    SpatialVelocity, 
    MakeRenderEngineGl, 
    LogVectorOutput, 
    GeometryInstance, 
    Box, 
    IllustrationProperties, 
    Rgba,
    StartMeshcat,
    ModelVisualizer,
    Sphere,
    PidController,
    ConstantVectorSource,
    SchunkWsgPositionController,
    InverseDynamicsController,
    MultibodyPlant,
    Parser,
    BodyIndex,
    JointIndex,
    HydroelasticContactRepresentation
)
from pydrake.common import RandomGenerator
from pydrake.common.yaml import yaml_load_typed
from pydrake.lcm import DrakeLcmParams
from pydrake.manipulation import (
    ApplyDriverConfigs,
    IiwaDriver,
    SchunkWsgDriver,
    ZeroForceDriver,
)
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.multibody.parsing import (
    ModelDirective,
    ModelDirectives,
    ProcessModelDirectives,
)
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    Simulator,
    SimulatorConfig,
)
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.lcm import ApplyLcmBusConfig
from pydrake.systems.sensors import (
    ApplyCameraConfig,
    CameraConfig,
)
from pydrake.visualization import (
    ApplyVisualizationConfig,
    VisualizationConfig,
)
from manipulation.scenarios import AddMultibodyTriad
# from bar_system import BarPositionDetector, SpecificBodyPoseExtractor
# from camera import PointCloudGeneration, add_rgbd_sensors
# from robot_commander import IK, wsg_2, MotionPlanner_wsg
from perception_mit_ring import PointCloudGenerator, TrajectoryPredictor, add_cameras
from grasp_select_ring import GraspSelector
# from grasp_select_bar import GraspSelector
from motion_planner import MotionPlanner
from graspnet_data_ring import GraspPredictor

@dc.dataclass
class Scenario:
    """Defines the YAML format for a (possibly stochastic) scenario to be
    simulated.
    """

    # Random seed for any random elements in the scenario. The seed is always
    # deterministic in the `Scenario`; a caller who wants randomness must
    # populate this value from their own randomness.
    random_seed: int = 0

    # The maximum simulation time (in seconds).  The simulator will attempt to
    # run until this time and then terminate.
    simulation_duration: float = math.inf

    # Simulator configuration (integrator and publisher parameters).
    simulator_config: SimulatorConfig = SimulatorConfig(
        max_step_size=1e-3, #5e-6, #,
        accuracy=1.0e-4, #1.0e-4,
        target_realtime_rate=1.0)

    # Plant configuration (time step and contact parameters).
    plant_config: MultibodyPlantConfig = MultibodyPlantConfig()
    plant_config.time_step = 0.00005

    # All of the fully deterministic elements of the simulation.
    directives: typing.List[ModelDirective] = dc.field(default_factory=list)

    # A map of {bus_name: lcm_params} for LCM transceivers to be used by
    # drivers, sensors, etc.
    lcm_buses: typing.Mapping[str, DrakeLcmParams] = dc.field(
        default_factory=lambda: dict(default=DrakeLcmParams()))

    # For actuated models, specifies where each model's actuation inputs come
    # from, keyed on the ModelInstance name.
    model_drivers: typing.Mapping[str, typing.Union[
        IiwaDriver,
        SchunkWsgDriver,
        ZeroForceDriver,
    ]] = dc.field(default_factory=dict)

    # Cameras to add to the scene (and broadcast over LCM). The key for each
    # camera is a helpful mnemonic, but does not serve a technical role. The
    # CameraConfig::name field is still the name that will appear in the
    # Diagram artifacts.
    cameras: typing.Mapping[str, CameraConfig] = dc.field(default_factory=dict)

    visualization: VisualizationConfig = VisualizationConfig()


def _load_scenario(*, filename, scenario_name, scenario_text):
    """Implements the command-line handling logic for scenario data.
    Returns a `Scenario` object loaded from the given input arguments.
    """
    result = yaml_load_typed(
        schema=Scenario,
        filename=filename,
        child_name=scenario_name,
        defaults=Scenario())
    result = yaml_load_typed(
        schema=Scenario,
        data=scenario_text,
        defaults=result)
    return result

# def launch_cylinder(plant, context, initial_pose, velocity, angle, roll, pitch, yaw):
#     # Convert angle to radians
#     angle_rad = np.radians(angle)
#     # Calculate initial velocity components

#     # Set the initial pose of the cylinder
#     cylinder_body = plant.GetBodyByName("noodle")
#     plant.SetFreeBodyPose(context, cylinder_body, initial_pose)
#     plant.SetFreeBodySpatialVelocity(plant.GetBodyByName("noodle"), 
#                                      SpatialVelocity(), 
#                                      context)

#     initial_velocity = np.array([velocity * np.cos(angle_rad), 0, velocity * np.sin(angle_rad)])

#     # Set the initial state
#     plant.SetFreeBodySpatialVelocity(plant.GetBodyByName("noodle"), 
#                                      SpatialVelocity(np.array([roll, pitch, yaw]), initial_velocity), 
#                                      context)

def launch_obj(plant, context, initial_pose, velocity, roll, pitch, yaw, obj):
    # Target point
    target_x = 1.5
    target_y = 0
    target = np.array([target_x, target_y, 2.0])
    
    # Extract the initial position from the initial pose
    initial_position = initial_pose.translation() #+ [0.5 , 0 , 0]
    
    # Calculate the direction vector from the initial position to the target
    direction_vector = target - initial_position
    # direction_vector[2] = 0  # Ensure the movement is in the xy-plane
    direction_normalized = direction_vector / np.linalg.norm(direction_vector)
    
    # Calculate the initial velocity components based on the direction vector
    initial_velocity = velocity * direction_normalized
    
    # Set the initial pose of the cylinder
    cylinder_body = plant.GetBodyByName(obj)
    plant.SetFreeBodyPose(context, cylinder_body, initial_pose)
    
    # Clear any existing velocity
    plant.SetFreeBodySpatialVelocity(plant.GetBodyByName(obj), SpatialVelocity(), context)
    
    # Calculate the orientation based on roll, pitch, yaw
    orientation = RollPitchYaw(np.radians(roll), np.radians(pitch), np.radians(yaw))#.ToRotationMatrix()
    
    # Update the initial pose with the new orientation
    # new_initial_pose = RigidTransform(orientation, initial_position)
    plant.SetFreeBodyPose(context, cylinder_body, initial_pose)
    
    # Set the initial velocity with the new direction
    plant.SetFreeBodySpatialVelocity(plant.GetBodyByName(obj), 
                                     SpatialVelocity(np.radians([roll,pitch,yaw]), initial_velocity), 
                                     context)


def generate_random_initial_pose():
    # Randomize initial position within specified ranges
    x_pos = np.random.uniform(-0.3, 0)  # Range for x: [-1, 0]
    y_pos = np.random.uniform(-1.2, 1.2) #np.random.uniform(-0.5, 0.5)  # Range for y: [-0.5, 0.5]
    z_pos = 0  # Fixed value for z

    # Randomize initial orientation within specified ranges
    # Convert degrees to radians for RollPitchYaw
    x_rot_deg = np.random.uniform(-5, 5)
    y_rot_deg = 90  # Range for y: [-5, 5]
    z_rot_deg = np.random.uniform(-14, 14)  # Range for z: [-14, 14]
    launching_position = [x_pos,y_pos,z_pos]
    launching_orientation = [x_rot_deg,y_rot_deg,z_rot_deg]
    print(f'launching position:{launching_position}, orientation:{launching_orientation}')
    # Convert degrees to radians
    x_rot_rad = np.radians(x_rot_deg)
    y_rot_rad = np.radians(y_rot_deg)
    z_rot_rad = np.radians(z_rot_deg)

    # Create RigidTransform with randomized position and orientation
    initial_pose = RigidTransform(
        RollPitchYaw(x_rot_rad, y_rot_rad, z_rot_rad),
        [x_pos, y_pos, z_pos]
    )

    return launching_position, launching_orientation, initial_pose

        
def run(*, scenario, graphviz=None, meshcat):
    """Runs a simulation of the given scenario.
    """

    # visualizer = ModelVisualizer(meshcat=meshcat)
    builder = DiagramBuilder()

    # Create the multibody plant and scene graph.
    sim_plant, scene_graph = AddMultibodyPlant(
        config=scenario.plant_config,
        builder=builder)

    # Add model directives.
    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        plant=sim_plant)

    #Random seed 
    grasp_random_seed = np.random.randint(0, 100000)
    np.random.seed(grasp_random_seed)
        
    #thrown model
    obj_name = 'ring' #noodle ring
    #thrown velocity
    velocity = np.random.uniform(low=5.0, high=6.0)
    print(f'velcocity:{velocity}')
    roll = 0
    pitch = np.random.uniform(low=-0.3, high=0.3)
    yaw = np.random.uniform(low=-0.3, high=0.3)

    #Add plane to visualize
    # plane_thickness = 0.01  
    # plane_width = 3 
    # plane_height = 3
    # plane_pose = RigidTransform(p=np.array([1.7, 0, 0.0]))
    # # box_pose = RigidTransform(RollPitchYaw(np.radians([0, 0, -270])).ToRotationMatrix() @ RollPitchYaw(np.radians([-150, 0, 0])).ToRotationMatrix(), [1.2192, 0, 2.0])
    # plane_shape = Box(plane_thickness, plane_width, plane_height)
    # # box_shape = Box(0.1, 0.2, 0.05) 
    # plane_instance = GeometryInstance(plane_pose, plane_shape, "yz_plane")
    # # box_instance = GeometryInstance(box_pose, box_shape, "cam_vis")
    # illustration_props = IllustrationProperties()
    # illustration_props.AddProperty("phong", "diffuse", Rgba(1.0, 1.0, 0.0, 0.0))  # Red color, fully opaque
    # plane_instance.set_illustration_properties(illustration_props)
    # # box_instance.set_illustration_properties(illustration_props)
    # source_id = sim_plant.get_source_id()
    # scene_graph.RegisterAnchoredGeometry(source_id, plane_instance)
    # # scene_graph.RegisterAnchoredGeometry(source_id, box_instance)
    # meshcat.SetObject('Box', plane_shape, rgba=Rgba(0.0, 1.0, 0.0))
    # meshcat.SetTransform('Box', plane_pose)
    AddMultibodyTriad(sim_plant.GetFrameByName(obj_name), scene_graph)
    # Now the plant is complete.
    sim_plant.Finalize()
    
    # Add LCM buses. (The simulator will handle polling the network for new
    # messages and dispatching them to the receivers, i.e., "pump" the bus.)
    lcm_buses = ApplyLcmBusConfig(
        lcm_buses=scenario.lcm_buses,
        builder=builder)
    # print(scenario.lcm_buses)
    # Add actuation inputs.
    ApplyDriverConfigs(
        driver_configs=scenario.model_drivers,
        sim_plant=sim_plant,
        models_from_directives=added_models,
        lcm_buses=lcm_buses,
        builder=builder)

    # Add scene cameras.
    for _, camera in scenario.cameras.items():
        ApplyCameraConfig(
            config=camera,
            builder=builder,
            lcm_buses=lcm_buses)

    # # Landing z Detection
    # detector = BarPositionDetector(target_x_position=1.2192, plant = sim_plant)  # Specify your target X position here
    # builder.AddSystem(detector)
    # builder.Connect(state.GetOutputPort('specific_body_pose'), detector.GetInputPort('noodle_state'))
    # z_log = LogVectorOutput(detector.GetOutputPort('z_at_target_x'), builder)
    # z_log.set_name("z_position_logger")

    # # Add camera system
    # cameras, camera_poses = add_rgbd_sensors(builder, sim_plant, scene_graph, camera_width = 800, camera_height = 600)
    # pointcloud_gen = PointCloudGeneration(sim_plant, cameras, camera_poses, meshcat)
    # builder.AddSystem(pointcloud_gen)
    # pointcloud_gen.ConnectCameras(builder, cameras)

    #object initial state setup
    initial_position = [-1.5, 0, 1]  # Example initial position y=[-0.5,0.5] z = 1 [-0.5, -0.5, 1]
    if obj_name == 'ring':
        launching_position, launching_orientation, initial_pose = generate_random_initial_pose() #for cylinder [90, 0, 14] ;
    elif obj_name == 'noodle':
        launching_position, launching_orientation, initial_pose = generate_random_initial_pose()


    #iiwa_1
    model_instance_iiwa_1 = sim_plant.GetModelInstanceByName('iiwa')
    
    #iiwa_2
    model_instance_iiwa_2 = sim_plant.GetModelInstanceByName('iiwa_2')

    #wsg_1_controller
    model_instance_wsg_1 = sim_plant.GetModelInstanceByName('wsg')
    wsg_1_controller = builder.AddSystem(
        SchunkWsgPositionController(
            kp_command=2500, kd_command=20,
            kp_constraint=2500, kd_constraint=20,
            default_force_limit=1000, time_step=0.001
        )
    )
    wsg_1_controller.set_name('wsg_1' + ".controller")
    builder.Connect(
        wsg_1_controller.get_generalized_force_output_port(),
        sim_plant.get_actuation_input_port(model_instance_wsg_1),
    )
    builder.Connect(
        sim_plant.get_state_output_port(model_instance_wsg_1),
        wsg_1_controller.get_state_input_port(),
    )

    #wsg_2_controller
    model_instance_wsg_2 = sim_plant.GetModelInstanceByName('wsg_2')
    wsg_2_controller = builder.AddSystem(
        SchunkWsgPositionController(
            kp_command=2500, kd_command=20,
            kp_constraint=2500, kd_constraint=20,
            default_force_limit=1000, time_step=0.001
        )
    )
    wsg_2_controller.set_name('wsg_2' + ".controller")
    builder.Connect(
        wsg_2_controller.get_generalized_force_output_port(),
        sim_plant.get_actuation_input_port(model_instance_wsg_2),
    )
    builder.Connect(
        sim_plant.get_state_output_port(model_instance_wsg_2),
        wsg_2_controller.get_state_input_port(),
    )

    # #Retrieve iiwa State
    body_poses_output_port = sim_plant.get_body_poses_output_port()

    ### Camera Setup
    icp_cameras, icp_camera_transforms = add_cameras(
        builder=builder,
        scene_graph = scene_graph ,
        plant=sim_plant,
        camera_width=800//2,
        camera_height=600//2,
        horizontal_num=4,
        vertical_num=5,
        camera_distance=7,
        cameras_center=[0, 0, 0],
        meshcat=meshcat
    )
    point_cloud_cameras_center = [0, 0, 0]
    point_cloud_cameras, point_cloud_camera_transforms = add_cameras(
        builder=builder,
        scene_graph =scene_graph ,
        plant=sim_plant,
        camera_width=800,
        camera_height=600,
        horizontal_num=8,
        vertical_num=4,
        camera_distance=1,
        cameras_center=point_cloud_cameras_center,
        meshcat=meshcat
    )

    ### Point Cloud Capturing Setup (origin)
    obj_point_cloud_system = builder.AddSystem(PointCloudGenerator(
        cameras=point_cloud_cameras,
        camera_transforms=point_cloud_camera_transforms,
        cameras_center=point_cloud_cameras_center,
        pred_thresh=5,
        thrown_model_name=obj_name,
        plant=sim_plant,
        meshcat=meshcat
    ))
    obj_point_cloud_system.ConnectCameras(builder, point_cloud_cameras)

    ### Moving pointcloud
    # realtime_point_cloud_system = builder.AddSystem(PointCloudGenerator(
    #     cameras=icp_cameras,
    #     camera_transforms=icp_camera_transforms,
    #     cameras_center=[0,0,0],
    #     pred_thresh=5,
    #     thrown_model_name=obj_name,
    #     plant=sim_plant,
    #     meshcat=meshcat
    # ))
    # realtime_point_cloud_system.ConnectCameras(builder, icp_cameras)

    ### Trajectory Prediction Setup
    traj_pred_system = builder.AddSystem(TrajectoryPredictor(
        cameras=icp_cameras,
        camera_transforms=icp_camera_transforms,
        pred_thresh=5,
        pred_samples_thresh=6,  # how many views of object are needed before outputting predicted traj
        thrown_model_name=obj_name,
        ransac_iters=20,
        ransac_thresh=0.01,
        ransac_rot_thresh=0.1,
        ransac_window=30,
        plant=sim_plant,
        estimate_pose= True, #("ball" not in obj_name),
        meshcat=meshcat,
        initial_pose = initial_pose
    ))
    traj_pred_system.ConnectCameras(builder, icp_cameras)
    builder.Connect(obj_point_cloud_system.GetOutputPort("point_cloud"), traj_pred_system.point_cloud_input_port)
    # builder.Connect(state.GetOutputPort('noodle_pose'), traj_pred_system.GetInputPort('noodle_state'))

    #Grasp Select 
    iiwa1_pose = RigidTransform(RollPitchYaw(np.radians([0, 0, -90])).ToRotationMatrix(),[2.3, 0.50, 0.0])
    iiwa2_pose = RigidTransform(RollPitchYaw(np.radians([0, 0, 90])).ToRotationMatrix(),[2.3, -0.50, 0.0])
    grasp_selector = builder.AddSystem(GraspSelector(sim_plant, scene_graph, meshcat, obj_name, grasp_random_seed, iiwa1_pose, iiwa2_pose))
    builder.Connect(traj_pred_system.GetOutputPort("object_trajectory"), grasp_selector.GetInputPort("object_trajectory"))
    builder.Connect(obj_point_cloud_system.GetOutputPort("point_cloud"), grasp_selector.GetInputPort("object_pc"))

    #Motion Planner
    motion_planner = builder.AddSystem(MotionPlanner(sim_plant, meshcat))
    builder.Connect(grasp_selector.GetOutputPort("grasp_selection"), motion_planner.GetInputPort("grasp_selection"))
    builder.Connect(body_poses_output_port, motion_planner.GetInputPort("iiwa_current_pose_1"))
    builder.Connect(body_poses_output_port, motion_planner.GetInputPort("iiwa_current_pose_2"))
    builder.Connect(traj_pred_system.GetOutputPort("object_trajectory"), motion_planner.GetInputPort("object_trajectory"))
    builder.Connect(sim_plant.get_state_output_port(model_instance_iiwa_1), motion_planner.GetInputPort("iiwa_state_1")) 
    builder.Connect(sim_plant.get_state_output_port(model_instance_iiwa_2), motion_planner.GetInputPort("iiwa_state_2")) 
    builder.Connect(motion_planner.GetOutputPort("wsg_command_1"), wsg_1_controller.get_desired_position_input_port())
    builder.Connect(motion_planner.GetOutputPort("wsg_command_2"), wsg_2_controller.get_desired_position_input_port())

    #inverse dynamics controller iiwa_1
    controller_plant = MultibodyPlant(time_step=0.001)
    controller_iiwa = Parser(controller_plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf")[0]  # ModelInstance object
    # controller_iiwa = AddIiwa(controller_plant)
    controller_plant.WeldFrames(controller_plant.world_frame(), controller_plant.GetFrameByName("base"))
    controller_plant.Finalize()
    num_iiwa_positions = controller_plant.num_positions(controller_iiwa)
    # for body_index in range(controller_plant.num_bodies()):
    #     body = controller_plant.get_body(BodyIndex(body_index) )
    #     body_name = body.name()
    #     num_positions_body = controller_plant.num_positions(body.model_instance())
    #     if num_positions_body > 0:
    #         print(f"Body '{body_name}' has {num_positions_body} positions.")

    # # Loop through all joints to see details about their positions
    # for joint_index in range(controller_plant.num_joints()):
    #     joint = controller_plant.get_joint(JointIndex(joint_index) )
    #     joint_name = joint.name()
    #     num_positions_joint = joint.num_positions()
    #     if num_positions_joint > 0:
    #         print(f"Joint '{joint_name}' contributes {num_positions_joint} positions.")
    iiwa_1_controller = builder.AddSystem(InverseDynamicsController(controller_plant, [1200]*num_iiwa_positions, [1]*num_iiwa_positions, [100]*num_iiwa_positions, True))
    builder.Connect(sim_plant.get_state_output_port(model_instance_iiwa_1), iiwa_1_controller.GetInputPort("estimated_state"))
    builder.Connect(motion_planner.GetOutputPort("iiwa_command_1"), iiwa_1_controller.GetInputPort("desired_state"))
    builder.Connect(motion_planner.GetOutputPort("iiwa_acceleration_1"), iiwa_1_controller.GetInputPort("desired_acceleration"))
    builder.Connect(iiwa_1_controller.GetOutputPort("generalized_force"), sim_plant.get_actuation_input_port(model_instance_iiwa_1))

    #inverse dynamics controller iiwa_2
    controller_plant_2 = MultibodyPlant(time_step=0.001)
    controller_iiwa_2 = Parser(controller_plant_2).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf")[0]  # ModelInstance object
    controller_plant_2.WeldFrames(controller_plant_2.world_frame(), controller_plant_2.GetFrameByName("base"))
    controller_plant_2.Finalize()
    num_iiwa_positions_2 = controller_plant_2.num_positions()
    iiwa_2_controller = builder.AddSystem(InverseDynamicsController(controller_plant_2, [1200]*num_iiwa_positions_2, [1]*num_iiwa_positions_2, [100]*num_iiwa_positions_2, True))
    builder.Connect(sim_plant.get_state_output_port(model_instance_iiwa_2), iiwa_2_controller.GetInputPort("estimated_state"))
    builder.Connect(motion_planner.GetOutputPort("iiwa_command_2"), iiwa_2_controller.GetInputPort("desired_state"))
    builder.Connect(motion_planner.GetOutputPort("iiwa_acceleration_2"), iiwa_2_controller.GetInputPort("desired_acceleration"))
    builder.Connect(iiwa_2_controller.GetOutputPort("generalized_force"), sim_plant.get_actuation_input_port(model_instance_iiwa_2))

    #add grasp net
    grasp_net = builder.AddSystem(GraspPredictor(sim_plant, scene_graph, obj_name, grasp_random_seed, velocity, roll, launching_position, launching_orientation, initial_pose, meshcat))
    builder.Connect(traj_pred_system.GetOutputPort("object_trajectory"), grasp_net.GetInputPort("object_trajectory"))
    builder.Connect(traj_pred_system.GetOutputPort("realtime_point_cloud"), grasp_net.GetInputPort("object_pc"))
    builder.Connect(grasp_selector.GetOutputPort("grasp_selection"), grasp_net.GetInputPort("grasp_selection"))
    builder.Connect(sim_plant.get_contact_results_output_port(), grasp_net.GetInputPort("contact_results_input"))

    # Add visualization.
    ApplyVisualizationConfig(scenario.visualization, builder, lcm_buses)

    # Build the diagram and its simulator.
    diagram = builder.Build()
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)

    # Sample the random elements of the context.
    random = RandomGenerator(scenario.random_seed)
    diagram.SetRandomContext(simulator.get_mutable_context(), random)
    context = simulator.get_mutable_context()
    plant_context = diagram.GetMutableSubsystemContext(sim_plant, context)
    # velocity = np.random.uniform(low=5.0, high=6.0)
    # print(f'velcocity:{velocity}')
    # # angle = np.random.uniform(low=20, high=50)
    # # angle = 24.6728821805643
    # # velocity = 3.45497304036041
    # roll = np.random.uniform(low=-0.3, high=0.3)
    # # pitch = np.random.uniform(low=0, high=0.2)
    # # yaw = np.random.uniform(low=-0.8, high=0.8)
    # # roll = 0.3 #-0.5534819644597283
    # pitch = 0 # 0.17857605884942562
    # yaw = 0 #0.342453262400656

    # Visualize the diagram, when requested.
    if graphviz is not None:
        with open(graphviz, "w", encoding="utf-8") as f:
            options = {"plant/split": "I/O"}
            f.write(diagram.GetGraphvizString(options=options))

    obj_point_cloud_system.CapturePointCloud(obj_point_cloud_system.GetMyMutableContextFromRoot(context))
    launch_obj(sim_plant, plant_context, initial_pose, velocity, roll, pitch, yaw, obj_name)#angle, roll, pitch, yaw)
    
    meshcat.StartRecording()
    simulator.AdvanceTo(0.86)
    meshcat.PublishRecording()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario_file", required=True,
        help="Scenario filename, e.g., "
             "drake/examples/hardware_sim/example_scenarios.yaml")
    parser.add_argument(
        "--scenario_name", required=True,
        help="Scenario name within the scenario_file, e.g., Demo in the "
             "example_scenarios.yaml; scenario names appears as the keys of "
             "the YAML document's top-level mapping item")
    parser.add_argument(
        "--scenario_text", default="{}",
        help="Additional YAML scenario text to load, in order to override "
             "values in the scenario_file, e.g., timeouts")
    parser.add_argument(
        "--graphviz", metavar="FILENAME",
        help="Dump the Simulator's Diagram to this file in Graphviz format "
             "as a debugging aid")
    args = parser.parse_args()
    # scenario = _load_scenario(
    #     filename=args.scenario_file,
    #     scenario_name=args.scenario_name,
    #     scenario_text=args.scenario_text)
    for i in range(2000):
        import tracemalloc
        tracemalloc.start()
        scenario = _load_scenario(
        filename=args.scenario_file,
        scenario_name=args.scenario_name,
        scenario_text=args.scenario_text)
        run(scenario=scenario, graphviz=args.graphviz, meshcat = StartMeshcat())
        print(i)
        gc.collect()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        for stat in top_stats[:10]:
            print(stat)
    # run(scenario=scenario, graphviz=args.graphviz)


if __name__ == "__main__":
    main()