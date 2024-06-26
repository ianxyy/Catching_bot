""" Miscellaneous Utility functions """
from enum import Enum
from typing import BinaryIO, Optional, Union, Tuple
from pydrake.all import (
    Diagram,
    RigidTransform,
    RotationMatrix,
    SpatialVelocity,
    MultibodyPlant,
    Context,
    AngleAxis,
)
from dataclasses import dataclass, field
from pydrake.common.yaml import yaml_load_typed
import numpy as np
import numpy.typing as npt
import pydot
import matplotlib.pyplot as plt


def diagram_update_meshcat(diagram, context=None) -> None:
    if context is None:
        context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)


def diagram_visualize_connections(diagram: Diagram, file: Union[BinaryIO, str]) -> None:
    """
    Create SVG file of system diagram.
    """
    if type(file) is str:
        file = open(file, "bw")
    graphviz_str = diagram.GetGraphvizString()
    svg_data = pydot.graph_from_dot_data(
        diagram.GetGraphvizString())[0].create_svg()
    file.write(svg_data)


def visualize_camera_plt(diagram: Diagram, camera_name: str, context=None, plt_show: bool = True) -> Optional[plt.Figure]:
    """
    Show Camera view using matplotlib.
    """
    if context is None:
        context = diagram.CreateDefaultContext()
    image = diagram.GetOutputPort(
        f"{camera_name}.rgb_image").Eval(context).data
    fig, ax = plt.subplots()
    ax.imshow(image)
    if not plt_show:
        return ax
    plt.show()


def throw_object_close(plant: MultibodyPlant, plant_context: Context, obj_name: str, obj_rot: RotationMatrix) -> None:
    """
    Version 1 (from further away, higher velocity)

    Move object to throwing position, generate initial velocity for object, then unfreeze its dynamics

    Args:
        plant: MultbodyPlant from hardware station
        plant_context: plant's context
        obj_name: string of the object's name in the scenario YAML (i.e. 'ycb2')
    """

    # Getting relevant data from plant
    model_instance = plant.GetModelInstanceByName(
        obj_name)  # ModelInstance object
    joint_idx = plant.GetJointIndices(model_instance)[0]  # JointIndex object
    joint = plant.get_joint(joint_idx)  # Joint object

    # Generate random object pose
    z = 0.3  # fixed z for now
    x = np.random.uniform(1.6, 1.7) #* np.random.choice([-1, 1])
    y = np.random.uniform(1.6, 1.7) #* np.random.choice([-1, 1])

    # Set object pose
    body_idx = plant.GetBodyIndices(model_instance)[0]  # BodyIndex object
    body = plant.get_body(body_idx)  # Body object
    pose = RigidTransform(obj_rot, [x, y, z])
    plant.SetFreeBodyPose(plant_context, body, pose)

    # Unlock joint so object is subject to gravity
    joint.Unlock(plant_context)

    v_magnitude = np.random.uniform(3.0, 3.1)#np.random.uniform(4.75, 5.0)
    angle_perturb = -np.random.uniform(0.18, 0.19) #* np.random.choice([-1, 1]) # must perturb by at least 0.1 rad to avoid throwing directly at iiwa
    # ensure the perturbation is applied such that it directs the obj away from iiwa
    if x * y > 0:  # x and y have same sign
        cos_alpha = x / np.sqrt(x**2 + y**2) + angle_perturb
        sin_alpha = y / np.sqrt(x**2 + y**2) - angle_perturb
    else:
        cos_alpha = x / np.sqrt(x**2 + y**2) + angle_perturb
        sin_alpha = y / np.sqrt(x**2 + y**2) + angle_perturb
    z_perturb = np.random.uniform(-0.1, 0.1)
    v_x = -v_magnitude * cos_alpha
    v_y = -v_magnitude * sin_alpha
    v_z = 3.7 + z_perturb

    # Define the spatial velocity
    spatial_velocity = SpatialVelocity(
        v=np.array([v_x, v_y, v_z]),  # m/s
        w=np.array([0, 0, 0]),  # rad/s
    )
    plant.SetFreeBodySpatialVelocity(
        context=plant_context, body=body, V_WB=spatial_velocity
    )


def throw_object_far(plant: MultibodyPlant, plant_context: Context, obj_name: str, obj_rot: RotationMatrix) -> None:
    """
    Version 1 (from further away, higher velocity)

    Move object to throwing position, generate initial velocity for object, then unfreeze its dynamics

    Args:
        plant: MultbodyPlant from hardware station
        plant_context: plant's context
        obj_name: string of the object's name in the scenario YAML (i.e. 'ycb2')
    """

    # Getting relevant data from plant
    model_instance = plant.GetModelInstanceByName(
        obj_name)  # ModelInstance object
    joint_idx = plant.GetJointIndices(model_instance)[0]  # JointIndex object
    joint = plant.get_joint(joint_idx)  # Joint object

    # Generate random object pose
    z = 0.5  # fixed z for now
    x = np.random.uniform(2.6, 2.7) * np.random.choice([-1, 1])
    y = np.random.uniform(2.6, 2.7) * np.random.choice([-1, 1])

    # Set object pose
    body_idx = plant.GetBodyIndices(model_instance)[0]  # BodyIndex object
    body = plant.get_body(body_idx)  # Body object
    pose = RigidTransform(obj_rot, [x, y, z])
    plant.SetFreeBodyPose(plant_context, body, pose)

    # Unlock joint so object is subject to gravity
    joint.Unlock(plant_context)

    v_magnitude = np.random.uniform(4.75, 5.0)
    angle_perturb = np.random.uniform(0.11, 0.12) * np.random.choice(
        [-1, 1]
    )  # must perturb by at least 0.1 rad to avoid throwing directly at iiwa
    # ensure the perturbation is applied such that it directs the obj away from iiwa
    if x * y > 0:  # x and y have same sign
        cos_alpha = x / np.sqrt(x**2 + y**2) + angle_perturb
        sin_alpha = y / np.sqrt(x**2 + y**2) - angle_perturb
    else:
        cos_alpha = x / np.sqrt(x**2 + y**2) + angle_perturb
        sin_alpha = y / np.sqrt(x**2 + y**2) + angle_perturb
    z_perturb = np.random.uniform(-0.5, 0.5)
    v_x = -v_magnitude * cos_alpha
    v_y = -v_magnitude * sin_alpha
    v_z = 3.8 + z_perturb

    # Define the spatial velocity
    spatial_velocity = SpatialVelocity(
        v=np.array([v_x, v_y, v_z]),  # m/s
        w=np.array([0, 0, 0]),  # rad/s
    )
    plant.SetFreeBodySpatialVelocity(
        context=plant_context, body=body, V_WB=spatial_velocity
    )


# def calculate_obj_distance_to_gripper(gripper_pose, obj_pose):
#     """
#     Calculates distance from object to gripper in gripper frame z-axis.
#     """
#     # Get distance in gripper frame z-axis from gripper y-axis ray object pose using vector projections
#     vector_gripper_to_obj = obj_pose.translation() - gripper_pose.translation()
#     vector_gripper_y_axis = gripper_pose.rotation().matrix()[:, 1]
#     projection_vector_gripper_to_obj_onto_vector_gripper_y_axis = (np.dot(vector_gripper_to_obj, vector_gripper_y_axis) / np.linalg.norm(vector_gripper_y_axis)) * vector_gripper_y_axis  # Equation for projection of one vector onto another
#     distance_vector = vector_gripper_to_obj - projection_vector_gripper_to_obj_onto_vector_gripper_y_axis
#     # Project distance only to gripper frame z-axis so that deviations in other axes don't affect the time at which the grippers close
#     vector_gripper_z_axis = gripper_pose.rotation().matrix()[:, 2]
#     projection_distance_vector_onto_gripper_frame_z_axis = (np.dot(distance_vector, vector_gripper_z_axis) / np.linalg.norm(vector_gripper_z_axis)) * vector_gripper_z_axis  # Equation for projection of one vector onto another
    
#     obj_distance_to_grasp = np.linalg.norm(projection_distance_vector_onto_gripper_frame_z_axis)

#     return obj_distance_to_grasp, vector_gripper_to_obj
    

def calculate_obj_distance_to_gripper(gripper_pose, obj_pose):
    """
    Calculates distance from object to gripper in gripper frame y-axis.
    """
    # Get distance in gripper frame z-axis from gripper y-axis ray object pose using vector projections
    vector_gripper_to_obj = obj_pose.translation() - gripper_pose.translation()
    vector_gripper_y_axis = gripper_pose.rotation().matrix()[:, 1]
    projection_vector_gripper_to_obj_onto_vector_gripper_y_axis = (np.dot(vector_gripper_to_obj, vector_gripper_y_axis) / np.linalg.norm(vector_gripper_y_axis))  # Equation for projection of one vector onto another
    obj_distance_to_grasp = np.abs(projection_vector_gripper_to_obj_onto_vector_gripper_y_axis)
 

    return obj_distance_to_grasp, vector_gripper_to_obj


@dataclass
class ObjectTrajectory:
    x: Tuple[np.float32, np.float32, np.float32] = (0, 0, 0)
    y: Tuple[np.float32, np.float32, np.float32] = (0, 0, 0)
    z: Tuple[np.float32, np.float32, np.float32] = (0, 0, 0)
    r: Tuple[AngleAxis, RotationMatrix] = field(default_factory=lambda: (AngleAxis(), RotationMatrix()))

    def __eq__(self, other):
        return np.allclose(
            [*self.x, *self.y, *self.z, self.r[0].angle(), *self.r[0].axis(), *self.r[1].matrix().flatten()],
            [*other.x, *other.y, *other.z, other.r[0].angle(), *other.r[0].axis(), *other.r[1].matrix().flatten()]
        )


    @staticmethod
    def _solve_single_traj(
            a: np.float32,
            x1: np.float32,
            t1: np.float32,
            x2: np.float32,
            t2: np.float32
        ) -> Tuple[np.float32, np.float32, np.float32]:
        return (a, *np.linalg.solve([[t1, 1], [t2, 1]], [x1 - a * t1 ** 2, x2 - a * t2 ** 2]))

    @staticmethod
    def _solve_rotation(
        r1_WO: RotationMatrix,
        t1: np.float32,
        r2_WO: RotationMatrix,
        t2: np.float32
    ) -> Tuple[AngleAxis, RotationMatrix]:
        w = (r1_WO.inverse() @ r2_WO).ToAngleAxis()
        rate = w.angle() / (t2 - t1)
        w.set_angle(rate)
        r0 = r1_WO @ RotationMatrix(AngleAxis(-rate * t1, w.axis()))
        return (w, r0)

    @staticmethod
    def CalculateTrajectory(
            X1: RigidTransform,
            t1: np.float32,
            X2: RigidTransform,
            t2: np.float32,
            g: np.float32 = 9.81,
        ) -> "ObjectTrajectory":
        p1 = X1.translation()
        p2 = X2.translation()
        r1_O = X1.rotation()
        r2_O = X2.rotation()
        return ObjectTrajectory(
            ObjectTrajectory._solve_single_traj(0, p1[0], t1, p2[0], t2),
            ObjectTrajectory._solve_single_traj(0, p1[1], t1, p2[1], t2),
            ObjectTrajectory._solve_single_traj(-g/2, p1[2], t1, p2[2], t2),
            ObjectTrajectory._solve_rotation(r1_O, t1, r2_O, t2)
        )

    def value(self, t: np.float32) -> RigidTransform:
        r = RotationMatrix(self.r[1] @ RotationMatrix(AngleAxis(self.r[0].angle() * t, self.r[0].axis())))
        return RigidTransform(r, [
            self.x[0] * t ** 2 + self.x[1] * t + self.x[2],
            self.y[0] * t ** 2 + self.y[1] * t + self.y[2],
            self.z[0] * t ** 2 + self.z[1] * t + self.z[2],
        ])

    def EvalDerivative(self, t: np.float32) -> npt.NDArray[np.float32]:
        return np.array([
            2 * self.x[0] * t + self.x[1],
            2 * self.y[0] * t + self.y[1],
            2 * self.z[0] * t + self.z[1]
        ] + list(self.r[0].axis() * self.r[0].angle()))