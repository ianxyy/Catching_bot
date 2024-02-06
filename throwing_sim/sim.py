import numpy as np
import os
import time
import csv
from pydrake.all import MultibodyPlant, SceneGraph, Simulator, SpatialVelocity, SpatialForce, MultibodyForces
from pydrake.common import temp_directory
from pydrake.geometry import StartMeshcat, Box, GeometryInstance, IllustrationProperties, Rgba#, MakePhongMaterial
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer
from multitask_net import MultiTaskNet
import torch
from sklearn.preprocessing import StandardScaler
from joblib import load

meshcat = StartMeshcat()
visualizer = ModelVisualizer(meshcat=meshcat)
bar_path = '/home/ece484/Catching_bot/throwing_sim/bar.sdf'
# visualizer.parser().AddModels(bar_path)

def create_scene(sim_time_step, plane_x_coord):
    # Clean up the Meshcat instance.
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=sim_time_step)
    parser = Parser(plant)
    # Define the dimensions of the plane
    plane_thickness = 0.01  # Thin thickness to represent a plane
    plane_width = 3  # Adjust the width and height as needed
    plane_height = 3
    # plane_x_coord = 0  # X-coordinate of the y-z plane
    # Define the pose of the plane
    plane_pose = RigidTransform(p=np.array([plane_x_coord, 0, 0]))

    # Create and add the plane geometry
    plane_shape = Box(plane_thickness, plane_width, plane_height)
    plane_instance = GeometryInstance(plane_pose, plane_shape, "yz_plane")
    # plane_instance.set_proximity_properties(prox_props)
    illustration_props = IllustrationProperties()
    illustration_props.AddProperty("phong", "diffuse", Rgba(1.0, 0.0, 0.0, 1.0))  # Red color, fully opaque
    plane_instance.set_illustration_properties(illustration_props)
    source_id = plant.get_source_id()
    scene_graph.RegisterAnchoredGeometry(source_id, plane_instance)
    # Loading models.
    # Load the table top and the cylinder we created.
    parser.AddModels(bar_path)

    # Finalize the plant after loading the scene.
    plant.Finalize()

    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    diagram = builder.Build()

    return diagram, plant

def initialize_simulation(diagram):
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.)
    return simulator


def check_intersection(plant, context, plane_x_coord):
    # Get the pose of the cylinder in the world frame
    cylinder_pose = plant.EvalBodyPoseInWorld(context, plant.GetBodyByName("noodle"))
    # print(cylinder_pose)
    # Define points near the ends of the cylinder in the cylinder's frame
    length = 1.4  # Length of the cylinder
    #imagine a cylinder is upright at (0,0,0) before rotate around x axis by 90
    point1 = np.array([0, 0, length / 2])  # Near top end
    point2 = np.array([0, 0, -length / 2])  # Near bottom end
    # Transform these points to the world frame
    world_point1 = cylinder_pose.multiply(point1) # be the right tip
    world_point2 = cylinder_pose.multiply(point2) # be the left tip
    # print('point_1:',world_point1)
    # print('point_2:',world_point2)
    # print('cylinder_pose:',cylinder_pose)
    # Check if either point is close to the y-z plane
    if world_point1[2] >= 0 and world_point2[2] >= 0 :
        if abs(world_point1[0] - plane_x_coord) < 0.0001 and abs(world_point2[0] - plane_x_coord) < 0.0001:
            coords = (world_point1,world_point2)
            print('both touched')
            return coords, 'both'
        elif abs(world_point1[0] - plane_x_coord) < 0.0001:   #0.0005
            # Record the z-coordinates of the points
            coords = (world_point1,world_point2)#(world_point1[2], world_point2[2])
            print('right touched')
            return coords, 'right'
        elif abs(world_point2[0] - plane_x_coord) < 0.0001:
            coords = (world_point1,world_point2)
            print('left touched')
            return coords, 'left'
        
        else:
            return None, None
    else:
        return False, None


def launch_cylinder(plant, context, initial_orientation, initial_position, velocity, angle, roll, pitch, yaw):
    # Convert angle to radians
    angle_rad = np.radians(angle)
    rpy = RollPitchYaw(np.radians(initial_orientation))
    initial_pose = RigidTransform(rpy.ToRotationMatrix(), initial_position)
    # Calculate initial velocity components

    # Set the initial pose of the cylinder
    cylinder_body = plant.GetBodyByName("noodle")
    plant.SetFreeBodyPose(context, cylinder_body, initial_pose)
    plant.SetFreeBodySpatialVelocity(plant.GetBodyByName("noodle"), 
                                     SpatialVelocity(), 
                                     context)

    initial_velocity = np.array([velocity * np.cos(angle_rad), 0, velocity * np.sin(angle_rad)])

    # Reset the simulation to initial state
    # simulator = Simulator(plant)
    # context = simulator.get_mutable_context()
    # plant_context = plant.GetMyMutableContextFromRoot(context)

    # Set the initial state
    plant.SetFreeBodySpatialVelocity(plant.GetBodyByName("noodle"), 
                                     SpatialVelocity(np.array([roll, pitch, yaw]), initial_velocity), 
                                     context)


def drag_sim(plant, context):
    cylinder_body = plant.GetBodyByName("noodle")
    velocity = plant.EvalBodySpatialVelocityInWorld(context, cylinder_body).translational()

    drag_coeff = 0.82
    air_density = 1.225 #kg/m^3 at sea level
    radius = 0.05
    length = 1.5

    cross_sectional_area = radius * length # side-on area

    drag_force_magnitude = 0.5 * drag_coeff * air_density * cross_sectional_area * (np.linalg.norm(velocity))**2
    drag_force = -drag_force_magnitude * velocity / np.linalg.norm(velocity)

    mass = plant.CalcTotalMass(context)
    acceleration_drag = drag_force / mass

    return acceleration_drag, velocity


def model_pred(model, velocity, angle, roll, pitch, yaw):
    new_data = np.array([velocity,angle,roll,pitch,yaw]).reshape(1,-1)
    scaler = load('scaler.joblib')
    X_normalized = scaler.transform(new_data)
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    with torch.no_grad():
        classification_predictions, regression_predictions = model(X_tensor)
        print('class',classification_predictions)
    classification_probs = torch.sigmoid(classification_predictions)
    # regression_predictions = scaler.inverse_transform(regression_predictions)
    print('classification_probs', classification_probs, 'regression_predictions', regression_predictions)


def run_simulation(sim_time_step, initial_orientation, initial_position, plane_x_coord):
    model = MultiTaskNet(input_size=5)
    model.load_state_dict(torch.load('model_state_dict.pth'))
    model.eval()
    with open('simulation_data4.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['TimeStamp', 'Velocity', 'Angle', 'Roll', 'Pitch', 'Yaw', 'AverageLeftZ', 'AverageRightZ', 'LeftTouchTime','RightTouchTime'])
        diagram , plant= create_scene(sim_time_step, plane_x_coord)
        simulator = initialize_simulation(diagram)
        # context = simulator.get_mutable_context()
        # plant_context = diagram.GetMutableSubsystemContext(plant, context)
        meshcat.StartRecording()
        # import pdb; pdb.set_trace()
        success_count = 0
        loop_count = 0
        for i in range(5000):
            loop_count += 1
            print('loop_count', loop_count)
            left_z_list = []
            right_z_list = []
            left_touch_time = -1.0
            right_touch_time = -1.0
            # Generate random velocity and angle
            touch_flag = False
            velocity_min = 4
            velocity_max = 10
            angle_min = 20
            angle_max = 85
            velocity = np.random.uniform(low=velocity_min, high=velocity_max)
            angle = np.random.uniform(low=angle_min, high=angle_max)
            # velocity = 8.3
            # angle = 37.3
            roll = np.random.uniform(low=-0.8, high=0.8)
            pitch = np.random.uniform(low=0, high=0.2)
            yaw = np.random.uniform(low=-0.8, high=0.8)
            # roll = 0.2
            # pitch = 0
            # yaw = -0.2
            model_pred(model, velocity, angle, roll, pitch, yaw)
            context = simulator.get_mutable_context()
            plant_context = diagram.GetMutableSubsystemContext(plant, context)
            print('angle:',angle)
            print('velocity:',velocity)
            # print('roll:', roll, 'pitch:', pitch, 'yaw:', yaw)
            # Set the initial pose and launch the cylinder

            launch_cylinder(plant, plant_context, initial_orientation, initial_position,velocity, angle, roll, pitch, yaw)
            # Reset the simulation state for the next launch
            context.SetTime(0)
            simulator.Initialize()
            time_stamp = time.time()
            # import pdb; pdb.set_trace()
            while context.get_time() < 3.0:
                acceleration_drag, curr_velocity = drag_sim(plant, plant_context)
                # Apply the drag force
                velocity_drag = acceleration_drag * sim_time_step
                new_velocity = curr_velocity + velocity_drag
                # print('current velocity:', curr_velocity)
                # print('dra_velocity:', velocity_drag)
                plant.SetFreeBodySpatialVelocity(
                    plant.GetBodyByName("noodle"), SpatialVelocity(np.array([roll, pitch, yaw]), new_velocity), plant_context
                )

                simulator.AdvanceTo(context.get_time() + sim_time_step)
                
                z_coords, touch_side = check_intersection(plant, plant_context, plane_x_coord)
                if z_coords is False:
                    break
                elif z_coords is not None:
                    touch_flag = True
                    if touch_side == 'both':
                        left_z_list.append(z_coords[1][2])
                        right_z_list.append(z_coords[0][2])
                    elif touch_side == 'left':
                        left_touch_time = time.time()
                        left_z_list.append(z_coords[1][2])
                    else:    
                        right_touch_time = time.time()
                        right_z_list.append(z_coords[0][2])
                    print("Intersection detected. left-coordinates:", z_coords[1], "right-coordinates:", z_coords[0])
                    # break
            average_left_z = np.average(left_z_list) if left_z_list else -1.0
            average_right_z = np.average(right_z_list) if right_z_list else -1.0
            if touch_flag:
                success_count += 1 
            # else:
            #     average_left_z = -1.0
            #     average_right_z = -1.0
            # print(left_z_list)
            # print(right_z_list)
            print('average_left:',average_left_z, 'average_right:',average_right_z) if touch_flag else print('didnt touch')
            writer.writerow([time_stamp, velocity, angle, roll, pitch, yaw, average_left_z, average_right_z, left_touch_time, right_touch_time])

            # if not touch_flag:
            #     print('didnt touch')
            time.sleep(1.0)
        print('rate', success_count/50)
    # Record the simulation in Meshcat
    # meshcat.StartRecording()
    # simulator.AdvanceTo(5.0)  # Simulate for 5 seconds
    # # meshcat.stop_recording()
    # meshcat.PublishRecording()




initial_position = [0, 0, 1]  # Example initial position
initial_orientation = [90, 0, 0]
# plant = create_scene(0.0001)
run_simulation(sim_time_step=0.000008, initial_orientation=initial_orientation, initial_position =initial_position, plane_x_coord = 3)