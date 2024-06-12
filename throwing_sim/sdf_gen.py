import math

num_segments = 300  # Number of segments to approximate the ring
inner_radius_x = 0.47  # Inner radius along x-axis
inner_radius_y = 0.35  # Inner radius along y-axis
outer_radius_x = 0.5  # Outer radius along x-axis
outer_radius_y = 0.38  # Outer radius along y-axis
height = 0.05  # Height of the ring

# Calculate the angle between each segment
angle_increment = 360.0 / num_segments

# Begin the SDF string for the model
sdf_content = """<?xml version='1.0'?>
<sdf version='1.6'>
    <model name='approximate_elliptical_ring'>
        <static>false</static>
        <link name='ring'>
            <inertial>
                <mass>1.0</mass>
                <inertia>
                    <ixx>0.1</ixx>
                    <iyy>0.1</iyy>
                    <izz>0.1</izz>
                </inertia>
            </inertial>
        </link>
"""

# Create each segment as a separate link
for i in range(num_segments):
    angle_degrees = i * angle_increment
    angle_radians = math.radians(angle_degrees)
    # Calculate the center position of the segment
    x = (inner_radius_x + outer_radius_x) / 2 * math.cos(angle_radians)
    y = (inner_radius_y + outer_radius_y) / 2 * math.sin(angle_radians)
    # Each segment is rotated around the z-axis
    sdf_content += f"""
        <link name='segment_{i}'>
            <pose>{x} {y} 0 0 0 {angle_radians}</pose>
            <collision name='collision'>
                <geometry>
                    <cylinder>
                        <radius>{(outer_radius_x - inner_radius_x) / 2}</radius>
                        <length>{height}</length>
                    </cylinder>
                </geometry>
            </collision>
            <visual name='visual'>
                <geometry>
                    <cylinder>
                        <radius>{(outer_radius_x - inner_radius_x) / 2}</radius>
                        <length>{height}</length>
                    </cylinder>
                </geometry>
            </visual>
        </link>
    """
sdf_joints = ""

# Loop to generate the joint definitions for each segment
for i in range(num_segments):
    sdf_joints += f"""
        <joint name='joint_segment_{i}' type='fixed'>
            <parent>ring</parent>
            <child>segment_{i}</child>
        </joint>
    """    

# Close the model tag
sdf_content += sdf_joints
sdf_content += """
    </model>
</sdf>
"""

# The sdf_content now contains the SDF definition for the elliptical ring
with open('/home/haonan/Catching_bot/throwing_sim/elliptical_ring_cylinder.sdf', "w") as file:
    file.write(sdf_content)
