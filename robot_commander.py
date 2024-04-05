"""A simple program to actuate the robot, an IIWA arm and a WSG gripper, with a
cyclic motion.
"""

import itertools
import os
import time
import sys
import numpy as np

from drake import lcmt_iiwa_command, lcmt_schunk_wsg_command
from pydrake.lcm import DrakeLcm
# from pydrake.all import 
# Use a non-default LCM url to communicate with the robot.
LCM_URL = "udpm://239.241.129.92:20185?ttl=0"

# The initial positions of the robot. This should be kept in sync with the
# scenario file's "Demo" example.
IIWA_Q0 = np.array([-2.09, 0.46, 0.78, -1.78, -0.2, -1.50, 1.3]) #[-0.2, 0.79, 0.32, -1.76, -0.36, 0.64, -0.73])
WSG_Q0 = 0.02  # In meters.

# The maximum deflection from IIWA and WSG's initial position, i.e., each joint
# will move between +/- MAX_DEFLECTION. The numbers are selected to avoid
# collisions within the scene.
IIWA_MAX_DEFLECTION = 0.4
WSG_MAX_DEFLECTION = 0.02

# The rate to send the robot commands.
COMMAND_HZ = 20

# How much time for a motion cycle, in seconds.
CYCLE_TIME = 10.0


def main_():
    lcm = DrakeLcm(LCM_URL)
    wsg_command = lcmt_schunk_wsg_command()
    wsg_command.force = 20.0
    iiwa_command = lcmt_iiwa_command()
    iiwa_command.num_joints = 7
    # for i in itertools.count():
    # sine = np.sin(2 * np.pi * i / (CYCLE_TIME * COMMAND_HZ))
    iiwa_command.joint_position = [0,0,0,0,0,0,0]#[0.67671456,  0.28651683, -0.55700998,  0.47087135,  1.70443827, 0.26305757, -0.02]
    # wsg_command.target_position_mm = 1e3 * (
        # WSG_Q0 + sine * WSG_MAX_DEFLECTION
    # )
    wsg_command.target_position_mm = -20 #150
    lcm.Publish(channel="IIWA_COMMAND", buffer=iiwa_command.encode())
    lcm.Publish(channel="SCHUNK_WSG_COMMAND", buffer=wsg_command.encode())
    # time.sleep(1 / COMMAND_HZ)
    # # When unit testing, just send one command.
    # if "TEST_SRCDIR" in os.environ:
    #     break


if __name__ == "__main__":
    main_()
