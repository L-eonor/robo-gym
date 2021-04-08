#!/usr/bin/env python3

from copy import deepcopy
import sys, math, copy, random
import numpy as np
from scipy.spatial.transform import Rotation as R
import gym
from gym import spaces
from gym.utils import seeding
from robo_gym.utils import utils, ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

class UR5RobotiqEnv(gym.Env):
    """Universal Robots UR5 base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.

    Attributes:
        ur5 (:obj:): Robot utilities object.
        observation_space (:obj:): Environment observation space.
        action_space (:obj:): Environment action space.
        distance_threshold (float): Minimum distance (m) from target to consider it reached.
        abs_joint_pos_range (np.array): Absolute value of joint positions range`.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    real_robot = False

    def __init__(self, rs_address=None, max_episode_steps=300, robotiq=85, **kwargs):
        #auxiliar objects
        self.ur5 = ur_utils.UR5ROBOTIQ(robotiq)
        self.ur_joint_dict=ur_utils.UR5ROBOTIQ(robotiq).ur_joint_dict
        self.robotiq=robotiq

        #joints
        #number of joints
        self.number_of_joints        = self.ur_joint_dict().get_number_of_joints()
        self.number_of_arm_joints    = self.ur_joint_dict().get_number_of_arm_joints()
        self.number_of_finger_joints = self.ur_joint_dict().get_number_of_finger_joints()
        #tol, max, min
        self.distance_threshold = 0.1
        self.abs_joint_pos_range = self.ur5.get_max_joint_positions()
        self.initial_joint_positions_low = self.ur_joint_dict()
        self.initial_joint_positions_high = self.ur_joint_dict()

        #simulation params
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
        self.seed()

        #observation space
        self.observation_space = self._get_observation_space()

        #action space
        self.action_space = self._get_action_space()
        
        # Initialize environment state
        self.state = env_state()


        self.last_position_on_success = []
        

        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, initial_joint_positions = None, ee_target_pose = None, type='random'):
        """Environment reset.

        Args:
            initial_joint_positions (list[7] or np.array[7]): robot joint positions in radians, standard order
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.last_action = None
        self.prev_base_reward = None
        
        #############################
        # Setting robot server state#
        #############################

        # Set initial robot joint positions, in standard order
        if initial_joint_positions:
            assert len(initial_joint_positions) == self.ur5.number_of_joint_positions
            ur5_initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            ur5_initial_joint_positions = self.last_position_on_success
        else:
            ur5_initial_joint_positions = self._get_initial_joint_positions()

        # update initial joint positions
        rs_state = server_state()
        rs_state.update_ur_joint_pos(self.ur_joint_dict().set_values_std_order(ur5_initial_joint_positions))
        
        # Set target End Effector pose
        if ee_target_pose:
            assert len(ee_target_pose) == self.rs_state__target_len #6
        else:
            ee_target_pose = self._get_target_pose()
        rs_state.update_target_pose(ee_target_pose)

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state = rs_state.get_server_message() )


        #############################
        # Reading robot server state#
        #############################
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")
        
        #Get current state and validade
        self.state, rs_state =self._get_current_env_state()
        
        # check if current position is in the range of the initial joint positions
        if (len(self.last_position_on_success) == 0) or (type=='random'):
            joint_positions=rs_state.get_state()["ur_j_pos"].get_values_std_order()
            tolerance = 0.1
            for joint in range(len(joint_positions)):
                if (joint_positions[joint]+tolerance < self.initial_joint_positions_low[joint]) or  (joint_positions[joint]-tolerance  > self.initial_joint_positions_high[joint]):
                    raise InvalidStateError('Reset joint positions are not within defined range')


        # go one empty action and check if there is a collision
        action = action_state().get_action_from_env_state(self.state) 
        _, _, done, info = self.step(action)
        self.elapsed_steps = 0
        if done:
            raise InvalidStateError('Reset started in a collision state')
            
        return self.state.to_array()

    def _reward(self, rs_state, action):
        return 0, False, {}

    def step(self, action):
        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action.to_array()), "%r (%s) invalid" % (action, type(action))

        # Convert environment action to Robot Server action
        rs_action = copy.deepcopy(action)
        # Scale action
        rs_action.values["ur_j_pos"].set_values_std_order(np.multiply(rs_action.values["ur_j_pos"].get_values_std_order(), self.abs_joint_pos_range.get_values_std_order() ) )
        # Convert action indexing from ur5 to ros
        rs_action = rs_action.values["ur_j_pos"].get_values_ros_order()
        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")

        #Get current state and validade
        self.state, rs_state =self._get_current_env_state()

        # Assign reward
        reward = 0
        done = False
        reward, done, info = self._reward(rs_state=rs_state, action=action)

        return self.state, reward, done, info

    def render():
        pass

    def _set_initial_joint_positions_range(self):
        '''
        joint positions order: shoulder_pan_joint, shoulder_lift_joint, elbow_joint, writ_1_joint, writ_2_joint, writ_3_joint, finger_joint (0 (open) as initial state)
        (updated the number of joint positions)
        '''
        self.initial_joint_positions_low = np.array([-0.65, -2.75, 1.0, -3.14, -1.7, -3.14, 0])
        self.initial_joint_positions_high = np.array([0.65, -2.0, 2.5, 3.14, -1.0, 3.14, 0])

    def _get_initial_joint_positions(self):
        """Generate random initial robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """
        self._set_initial_joint_positions_range()
        # Random initial joint positions
        joint_positions = np.random.default_rng().uniform(low=self.initial_joint_positions_low, high=self.initial_joint_positions_high)

        return joint_positions

    def _get_target_pose(self):
        """Generate target End Effector pose.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """

        return self.ur5.get_random_workspace_pose()

    def _get_observation_space(self):
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """

        # Joint position range tolerance
        pos_tolerance = np.full(self.ur5.number_of_joint_positions,0.1)
        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(np.full(self.number_of_joints, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(self.number_of_joints, -1.0), pos_tolerance)
        # Target coordinates range
        target_range = np.full(3, np.inf)
        # Joint positions range tolerance
        vel_tolerance = np.full(self.number_of_joints,0.5)
        # Joint velocities range used to determine if there is an error in the sensor readings
        max_joint_velocities = np.add(self.ur5.get_max_joint_velocities().get_values_std_order(), vel_tolerance)
        min_joint_velocities = np.subtract(self.ur5.get_min_joint_velocities().get_values_std_order(), vel_tolerance)
        # Definition of environment observation_space
        max_obs = np.concatenate((target_range, max_joint_positions, max_joint_velocities))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_joint_velocities))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _get_action_space(self):
        action_space_dim = self.ur5.number_of_joint_velocities
        #self.action_space = spaces.Dict(dict(
        #    arm_joints=spaces.Box(low=np.full((self.number_of_arm_joints), -1.0), high=np.full((self.number_of_arm_joints), 1.0), dtype=np.float32),
        #    finger_joints=spaces.Discrete (2) #0-open; 1-close
        #))
        action_space = spaces.Box(low=np.full((self.ur5.number_of_joint_velocities), -1.0), high=np.full((self.ur5.number_of_joint_velocities), 1.0), dtype=np.float32)
        return action_space

    def _get_current_env_state(self):
        """Requests the current robot state (simulated or real)

        Args:
            NaN

        Returns:
            new_state (env_state): Current state in environment format.
            rs_state (server_state): State in Robot Server format.

        """

        # Get Robot Server state
        rs_state=server_state()
        rs_state.set_server_from_message(np.nan_to_num(np.array(self.client.get_state_msg().state)))

        # Check if the length of the Robot Server state received is correct
        #if not len(rs_state)== self._get_robot_server_state_len():
        #    raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        new_state = rs_state.server_state_to_env_state(robotiq=self.robotiq)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(new_state.to_array() ):
            raise InvalidStateError()

        return new_state, rs_state

class env_state():
    """
    Encapsulates the environment state
    Includes:
        * target (np.array) pose in polar coordinates 
        * ur_j_pos (ur_joint_dict)-> robots' joint angles in a ur_joint_dict 
        * ur_j_vel (ur_joint_dict) -> robots' joint velocities in a ur_joint_dict

    """
    
    def __init__(self):
        """
        Populates the structure with:
            * target_polar (np.array)      : target pose in polar coordinates
            * ur_j_pos     (ur_joint_dict) : joint positions in angles (with zeros)
            * ur_j_vel     (ur_joint_dict) : joint velocities (with zeros)
        """
        self.state={
            "target_polar": np.zeros(3, dtype=np.float32),
            "ur_j_pos": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ur_j_vel": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
        }
    
    def update_target_polar(self, target_polar):
        """
        Updates the target coordinates:
        
        Args:
            target_polar (array like): new target point in polar coordinates

        Returns:
            none
        """

        self.state["target_polar"]=np.array(target_polar)

    def update_ur_j_pos(self, ur_joint_state):
        """
        Updates the joints' positions in angles :
        
        Args:
            ur_joint_state (ur_joint_dict): Joint position object, new values

        Returns:
            none
        """
        
        self.state["ur_j_pos"]=copy.deepcopy(ur_joint_state)

    def update_ur_j_vel(self, ur_joint_state):
        """
        Updates the joints' velocities :
        
        Args:
            ur_joint_state (ur_joint_dict): Joint vel object, new values

        Returns:
            none
        """
        
        self.state["ur_j_vel"]=copy.deepcopy(ur_joint_state)

    def to_array(self):
        """
        Retrieves the current state as a list. The order is: (target_polar + ur_j_pos + ur_j_vel )
        The ur_j_pos and ur_j_vel are displayed in standard order (from base to end effector)
        
        Args:
            None

        Returns:
            env_array (list): ordered list containing the current environment's state. The array includes the following: target_polar + ur_j_pos (std order) + ur_j_vel (std_order)
        """

        env_array= self.state["target_polar"].tolist() + self.state["ur_j_pos"].get_values_std_order().tolist() + self.state["ur_j_vel"].get_values_std_order().tolist() 

        return env_array

class action_state():
    """
    Encapsulates an action
    Includes: 
        * ur_j_pos (ur_joint_dict) -> robots' joint angles in a ur_joint_dict
    """
    
    def __init__(self):
        """
        Populates the structure with:
            * ur_j_pos     (ur_joint_dict) : joint positions in angles (with zeros)
        """

        self.values={
            "ur_j_pos": ur_utils.UR5ROBOTIQ().ur_joint_dict()
        }

    def get_action_from_env_state(self, env_state):
        """
        Updates the action values (joint angles)
        
        Args:
            env_state (env_state object): current environment's state

        Returns:
            self (So that: new_action=action_state().get_action_from_env_state(env_state) )
        """

        self.values["ur_j_pos"]=copy.deepcopy(env_state.state["ur_j_pos"])
        return self

    def to_array(self):
        """
        Retrieves the action as a list. The order is: (ur_j_pos  )
        The ur_j_pos  are displayed in standard order (from base to end effector)
        
        Args:
            None

        Returns:
            action_array (list): ordered list containing the current action description. The array includes the following: ur_j_pos (std order)
        """

        action_array= self.values["ur_j_pos"].get_values_std_order().tolist() 

        return action_array

class server_state():
    """
    Encapsulates the robot server state
    Includes:
        * target pose in [xyzrpy]
        * ur_j_pos-> robots' joint angles in a ur_joint_dict
        * ur_j_vel-> robots' joint velocities in a ur_joint_dict

        * ee_base_transform-> array len=7
        * collision-> array len=1

    """
    
    def __init__(self):
        """
        Populates the structure with:
            * target_xyzrpy     (np.array)      : target pose in xyzrpy
            * ur_j_pos          (ur_joint_dict) : joint positions in angles (with zeros)
            * ur_j_vel          (ur_joint_dict) : joint velocities (with zeros)
            * ee_base_transform (np.array)      : end effector base transform
            * collision         (np.array)      : collision array
        """
        
        self.state={
            "target_xyzrpy": np.zeros(6, dtype=np.float32), #xyzrpy
            "ur_j_pos": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ur_j_vel": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ee_base_transform": np.zeros(7, dtype=np.float32),
            "collision": np.zeros(1, dtype=np.float32)
        }

    def get_state(self):
        """
        Retrieves the state dictionary
        
        Args:
            None

        Returns:
            self.state (dictionary)-> dictionary object with current robot server state
        """

        return self.state

    def get_server_message(self):
        """
        Creates a message (list) to send to the server ( for setting the state). The order is: (target_xyzrpy + ur_j_pos + ur_j_vel + ee_base_transform + collision)
        The ur_j_pos and ur_j_vel are displayed in standard order (from base to end effector)
        
        Args:
            None

        Returns:
            msg (list)-> ordered list containing the server's state. The array includes the following: target_xyzrpy + ur_j_pos (std order) + ur_j_vel (std_order) + ee_base_transform + collision
        """

        msg= self.state["target_xyzrpy"].tolist() + self.state["ur_j_pos"].get_values_ros_order().tolist() + self.state["ur_j_vel"].get_values_ros_order().tolist() + self.state["ee_base_transform"].tolist() + self.state["collision"].tolist()
        
        return msg

    def update_target_pose(self, new_target_pose):
        """
        Updates the target coordinates:
        
        Args:
            new_target_pose (array like): new target point in xyzrpy

        Returns:
            none
        """

        self.state["target_xyzrpy"]=np.array(new_target_pose)

    def update_ur_joint_pos(self, new_joint_pos):
        """
        Updates the joints' positions in angles :
        
        Args:
            new_joint_pos (ur_joint_dict): Joint position object, new values

        Returns:
            none
        """

        self.state["ur_j_pos"]=copy.deepcopy(new_joint_pos)

    def update_ur_joint_vel(self, new_joint_vel):
        """
        Updates the joints' velocities:
        
        Args:
            new_joint_pos (ur_joint_dict): Joint vel object, new values

        Returns:
            none
        """

        self.state["ur_j_vel"]=copy.deepcopy(new_joint_vel)
    
    def update_ee_base_transform(self, new_ee_base_transform):
        """
        Updates the end effector base transform
        
        Args:
            new_ee_base_transform (array like): new effector base transform

        Returns:
            none
        """

        self.state["ee_base_transform"]=np.array(new_ee_base_transform)
    
    def update_collision(self, new_collision_state):
        """
        Updates the collision array
        
        Args:
            new_collision_state (array like): new collision info

        Returns:
            none
        """
        self.state["collision"]=np.array(new_collision_state)

    def set_server_from_message(self, msg):
        """
        Updates the state values: target, position, velocity, ee base transform and collition. Uses the info retrieved from the server by the corresponding message array
        
        Args:
            msg (list): server's state to be saved. Includes
               * the target_pose in xyzrpy coordinates
               * joint's position (angles) in ros order (alphabetical)-> elbow_joint, finger_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint
               * joint's velocities in ros order (alphabetical)-> elbow_joint, finger_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint
               * the end effector base transform
               * collision info

        Returns:
            None
        """

        #computes the list indexes here to make the code easily readable
        a= 0
        b= a + len(self.state["target_xyzrpy"])
        c= b + len(self.state["ur_j_pos"].joints)
        d= c + len(self.state["ur_j_vel"].joints)
        e= d + len(self.state["ee_base_transform"])
        f= e + len(self.state["collision"])

        #coppies info in the appropriate format
        self.update_target_pose(       msg[ a:b ] )
        self.update_ur_joint_pos(      ur_utils.UR5ROBOTIQ().ur_joint_dict().set_values_ros_order(msg[ b:c ]) )
        self.update_ur_joint_vel(      ur_utils.UR5ROBOTIQ().ur_joint_dict().set_values_ros_order(msg[ c:d ] ))
        self.update_ee_base_transform (msg[ d:e ] )
        self.update_collision(         msg[ e:f ] )
        
    def server_state_to_env_state(self, robotiq=85):
        """
        Creates the environment's state object based on the server state object. This means updating the environment state
        
        Args:
            robotiq (int): (85 or 140) reference to gripper model, in order to choose the appropriate joint limits

        Returns:
            new_env_state (env_state) -> new env state object with updated values
        """

        new_env_state=env_state()

        # Transform cartesian coordinates of target to polar coordinates 
        # with respect to the end effector frame
        target_coord = np.nan_to_num(self.state["target_xyzrpy"])[0:3]
        ee_base_transform =   np.nan_to_num(self.state["ee_base_transform"])
        
        ee_to_base_translation = ee_base_transform[0:3]
        ee_to_base_quaternion = ee_base_transform[3:8]
        ee_to_base_rotation = R.from_quat(ee_to_base_quaternion)
        base_to_ee_rotation = ee_to_base_rotation.inv()
        base_to_ee_quaternion = base_to_ee_rotation.as_quat()
        base_to_ee_translation = - ee_to_base_translation

        target_coord_ee_frame = utils.change_reference_frame(target_coord,base_to_ee_translation,base_to_ee_quaternion)
        target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)


        ##update
        new_env_state.update_target_polar(target_polar)
        ur_j_pos_norm=ur_utils.UR5ROBOTIQ(robotiq).normalize_ur_joint_dict(joint_dict=self.state["ur_j_pos"])
        new_env_state.update_ur_j_pos(ur_j_pos_norm)
        new_env_state.update_ur_j_vel(self.state["ur_j_vel"])

        return new_env_state



class GraspObjectUR5(UR5RobotiqEnv):
    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}

        # Calculate distance to the target
        target_coord = np.array(rs_state.state["target_xyzrpy"])[0:3]
        ee_coord =   np.array(rs_state.state["ee_base_transform"])[0:3]
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)

        # Reward base
        reward = -1 * euclidean_dist_3d
        
        joint_positions_normalized=ur_utils.UR5ROBOTIQ(self.robotiq).normalize_ur_joint_dict(joint_dict=rs_state.state["ur_j_pos"])

        delta = np.abs(np.subtract(joint_positions_normalized.get_values_std_order(), action.values["ur_j_pos"].get_values_std_order() ))
        reward = reward - (0.05 * np.sum(delta))

        if euclidean_dist_3d <= self.distance_threshold:
            reward = 100
            done = True
            info['final_status'] = 'success'
            info['target_coord'] = target_coord

        # Check if robot is in collision
        if rs_state.state["collision"] [-1] == 1:
            collision = True
        else:
            collision = False

        if collision:
            reward = -400
            done = True
            info['final_status'] = 'collision'
            info['target_coord'] = target_coord

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
            info['target_coord'] = target_coord

        return reward, done, info
    

class GraspObjectUR5Sim(GraspObjectUR5, Simulation):
    #cmd = "roslaunch ur_robot_server ur5Robotiq_sim_robot_server.launch \
    #    max_velocity_scale_factor:=0.2 \
    #    action_cycle_rate:=20"


    cmd = "roslaunch ur_robot_server ur5Robotiq_sim_robot_server.launch \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=true \
        gazebo_gui:=true"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        GraspObjectUR5.__init__(self, rs_address=self.robot_server_ip, robotiq=85, **kwargs)