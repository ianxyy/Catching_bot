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
from robot_commander import main_

from pydrake.all import RigidTransform, RollPitchYaw, RgbdSensor, SpatialVelocity, MakeRenderEngineGl, Box, GeometryInstance, IllustrationProperties, Rgba, StartMeshcat, Sphere,Parser
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
        max_step_size=1e-3,
        accuracy=1.0e-2,
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


def run(*, scenario, graphviz=None):
    """Runs a simulation of the given scenario.
    """
    builder = DiagramBuilder()
    meshcat = StartMeshcat()
    # Create the multibody plant and scene graph.
    sim_plant, scene_graph = AddMultibodyPlant(
        config=scenario.plant_config,
        builder=builder)

    # Add model directives.
    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        plant=sim_plant)
    
    #Add plane to visualize
    plane_thickness = 0.01  
    plane_width = 3 
    plane_height = 3
    plane_pose = RigidTransform(p=np.array([1.2192, 0, 0.0]))
    box_pose = RigidTransform(RollPitchYaw(np.radians([-90, 0, 180])).ToRotationMatrix(), [0.5, 1.6, 1.0])#(RollPitchYaw(np.radians([-110, 0, 180])).ToRotationMatrix(), [0.5, 1.6, 1.5])
    plane_shape = Box(plane_thickness, plane_width, plane_height)
    box_shape = Box(0.1, 0.2, 0.05) 
    plane_instance = GeometryInstance(plane_pose, plane_shape, "yz_plane")
    box_instance = GeometryInstance(box_pose, box_shape, "cam_vis")
    sphere_instance = GeometryInstance(RigidTransform(p=np.array([2, -1, 0.1])), Sphere(0.3), "sphere")
    illustration_props = IllustrationProperties()
    illustration_props.AddProperty("phong", "diffuse", Rgba(1.0, 1.0, 0.0, 1.0))  # Red color, fully opaque
    plane_instance.set_illustration_properties(illustration_props)
    box_instance.set_illustration_properties(illustration_props)
    sphere_instance.set_illustration_properties(illustration_props)
    source_id = sim_plant.get_source_id()
    scene_graph.RegisterAnchoredGeometry(source_id, plane_instance)
    scene_graph.RegisterAnchoredGeometry(source_id, box_instance)
    scene_graph.RegisterAnchoredGeometry(source_id, sphere_instance)
    # j_iiwa_2 = Parser(sim_plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision_2.urdf")[0]  # ModelInstance object
    # gripper_frame = sim_plant.GetFrameByName("iiwa_link_7", j_iiwa_2)
    # initial_pose_iiwa1 = RigidTransform(RollPitchYaw(np.radians([0, 0, 180])).ToRotationMatrix(), [0, 0.1, 0.1])
    # sim_plant.WeldFrames(sim_plant.world_frame(), sim_plant.GetFrameByName("base", j_iiwa_2), initial_pose_iiwa1)
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
    camera_properties = CameraConfig()
    camera_properties.z_far= 30
    camera_properties.z_near= 0.1
    camera_properties.width = 640
    camera_properties.height = 480
    camera_properties.show_rgb = True
    # camera_properties.lcm_bus = 'udpm://239.241.129.92:20185?ttl=0'
    camera_properties.focal = camera_properties.FovDegrees(x = None,y = 90)
    if not scene_graph.HasRenderer(camera_properties.renderer_name):
        scene_graph.AddRenderer(
            camera_properties.renderer_name, MakeRenderEngineGl())
    _, depth_camera = camera_properties.MakeCameras()
    camera_pose = box_pose #RigidTransform(RollPitchYaw(np.radians([-120, 5, 125])), [1.5, 0.8, 1.25])  # Adjust as needed. (0, np.pi/2, 0)
    rgbd_sensor = RgbdSensor(parent_id=scene_graph.world_frame_id(), X_PB=camera_pose, depth_camera=depth_camera, show_window = False )
    builder.AddSystem(rgbd_sensor)
    builder.Connect(scene_graph.get_query_output_port(), rgbd_sensor.query_object_input_port())

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
    # sim_plant.SetPositions(plant_context, j_iiwa_2, [0,0,0,0,0,0,0])
    # sim_plant.SetFreeBodySpatialVelocity(sim_plant.GetBodyByName("noodle"), 
    #                                 SpatialVelocity(np.array([0, 0, 0]), [3,0,5]), 
    #                                 plant_context)
    rgbd_sensor_context = rgbd_sensor.GetMyMutableContextFromRoot(context)
    color_image = rgbd_sensor.color_image_output_port().Eval(rgbd_sensor_context).data#[::-1]
    depth_image = rgbd_sensor.depth_image_32F_output_port().Eval(rgbd_sensor_context).data#[::-1]
    label_image = rgbd_sensor.label_image_output_port().Eval(rgbd_sensor_context).data
    plt.imshow(label_image)
    plt.show()
    print(sim_plant.get_body_poses_output_port())
    # Visualize the diagram, when requested.
    if graphviz is not None:
        with open(graphviz, "w", encoding="utf-8") as f:
            options = {"plant/split": "I/O"}
            f.write(diagram.GetGraphvizString(options=options))

    # Simulate.
    # while context.get_time() < 3.0:
    #     simulator.AdvanceTo(context.get_time() + 1.0)#(scenario.simulation_duration)
    #     color_image = rgbd_sensor.color_image_output_port().Eval(rgbd_sensor_context).data#[::-1]
    #     depth_image = rgbd_sensor.depth_image_32F_output_port().Eval(rgbd_sensor_context).data#[::-1]
    #     plt.imshow(color_image)
    #     plt.show()
                    
    X_WE = sim_plant.CalcRelativeTransform(plant_context, frame_A= sim_plant.world_frame(), frame_B=sim_plant.GetFrameByName("wsg_on_iiwa"))

    # Extract position and orientation from the transform
    position = X_WE.translation()
    rotation = X_WE.rotation()
    rpy = RollPitchYaw(rotation).vector()
    print(f"End-effector position: {position}")
    print(f"End-effector orientation (Roll, Pitch, Yaw): {rpy}")
    initial_pose = RigidTransform(RollPitchYaw(np.radians([0, 0, 0])).ToRotationMatrix(), [0, -0.3, 3.5])
    cylinder_body = sim_plant.GetBodyByName("ring")
    sim_plant.SetFreeBodyPose(plant_context, cylinder_body, initial_pose)
    # initial_pose = RigidTransform(RollPitchYaw(np.radians([0, 0, 0])).ToRotationMatrix(), [0, 0, 0.1])
    # cube = sim_plant.GetBodyByName("cube")
    # sim_plant.SetFreeBodyPose(plant_context, cube, initial_pose)
    main_()
    meshcat.StartRecording()
    simulator.AdvanceTo(2.0) #(scenario.simulation_duration)
    meshcat.PublishRecording()

    contact_results = sim_plant.get_contact_results_output_port().Eval(plant_context)
    print(f'contact:{contact_results.num_point_pair_contacts()}')
    for i in range(contact_results.num_point_pair_contacts()):
        point_pair_contact_info = contact_results.point_pair_contact_info(i)
        # Here you can access detailed information about the contact
        print(f"Contact force: {point_pair_contact_info.contact_force()}")
        print(f"Contact point: {point_pair_contact_info.contact_point()}")
        print(f"Body A: {sim_plant.GetBodyFromFrameId(sim_plant.GetBodyFrameIdOrThrow(point_pair_contact_info.bodyA_index())).name()}")
        print(f"Body B: {sim_plant.GetBodyFromFrameId(sim_plant.GetBodyFrameIdOrThrow(point_pair_contact_info.bodyB_index())).name()}")
    # meshcat.PublishRecording()
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("RGB Image")
    # plt.imshow(color_image)
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.title("Depth Image")
    # plt.imshow(depth_image, cmap='gray')
    # plt.axis('off')
    # plt.show()


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
    scenario = _load_scenario(
        filename=args.scenario_file,
        scenario_name=args.scenario_name,
        scenario_text=args.scenario_text)
    run(scenario=scenario, graphviz=args.graphviz)


if __name__ == "__main__":
    main()
