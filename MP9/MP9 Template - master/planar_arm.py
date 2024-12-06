import math, numpy as np
from matplotlib.axes import Axes

# An arm consists of a set of joints, each of which is at a certain angle relative to the previous
# Each joint connects between limbs of certain length
class Arm:
    def __init__(self, arm_lengths):
        # arm_lengths is the length of each arm limb
        self.arm_lengths = arm_lengths
        self.num_joints = len(arm_lengths)
        # each joint is limited in the angles it can take, we set this from -90 to 90 degrees
        self.min_angle = np.zeros(self.num_joints) - 90
        self.max_angle = np.zeros(self.num_joints) + 90
        # start_point is the starting (x,y) coordinate of the first limb/joint
        self.start_point = [0,0]
        
    # workspace_config is the (x,y) coordinate of each joint, i.e., forward kinematics
    def forward_kinematics(self, config):
        x = self.start_point[0]
        y = self.start_point[1]
        workspace_config = np.zeros((self.num_joints + 1, 2))
        workspace_config[0] = [x,y]
        rel_angle = 0
        for index, (angle, length) in enumerate(zip(config, self.arm_lengths)): 
            rel_angle += angle
            x += length * math.cos(math.radians(rel_angle)) 
            y -= length * math.sin(math.radians(rel_angle))
            workspace_config[index+1] = [x,y]
        return workspace_config
                  
    # Check if a configuration is within angle limits
    def in_boundary(self, config):
        for angle in config:
            for min_a, max_a in zip(self.min_angle, self.max_angle):
                if angle < min_a or angle > max_a:
                    return False
        return True
    
    def draw_space(self, ax : Axes):
        ax.set_xlim(-sum(self.arm_lengths), sum(self.arm_lengths))
        ax.set_ylim(-sum(self.arm_lengths), sum(self.arm_lengths))
        ax.set_aspect('equal')

    def draw_config(self, ax : Axes, config, **kwargs):
        workspace_config = self.forward_kinematics(config)
        ax.plot(workspace_config[:,0], workspace_config[:,1], **kwargs)
