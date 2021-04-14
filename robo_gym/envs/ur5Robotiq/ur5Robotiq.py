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

    def reset(self, initial_joint_positions = None, ee_target_pose = None, cube_destination_pose = None, type='random'):
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

        # Set target cube destination pose
        if cube_destination_pose:
            assert len(cube_destination_pose) == self.rs_state__destination_len #6
        else:
            cube_destination_pose = self._get_cube_destination()
        self.cube_destination_pose = cube_destination_pose

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state = rs_state.get_server_message() )

        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")
        
        #############################
        # Reading robot server state#
        #############################

        #Get current state, update obs space with cubes and validate
        self.state, rs_state =self._get_current_env_state_and_update_observation_space()

        # check if current position is in the range of the initial joint positions
        if (len(self.last_position_on_success) == 0) or (type=='random'):
            joint_positions=rs_state.get_state()["ur_j_pos"].get_values_std_order()
            tolerance = 0.1
            for joint in range(len(joint_positions)):
                if (joint_positions[joint]+tolerance < self.initial_joint_positions_low[joint]) or  (joint_positions[joint]-tolerance  > self.initial_joint_positions_high[joint]):
                    raise InvalidStateError('Reset joint positions are not within defined range')


        # go one empty action and check if there is a collision
        action = action_state().get_action_from_env_state(self.state) 
        _, _, done, info = self.step(action.values)
        self.elapsed_steps = 0
        if done:
            raise InvalidStateError('Reset started in a collision state')
            
        return self.state.to_array()

    def _reward(self, rs_state, action):
        return 0, False, {}

    def step(self, action):
        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action ), "%r (%s) invalid" % (action, type(action))

        # Convert environment action to Robot Server action
        rs_action = action_state() #copy.deepcopy(action)
        # Scale action
        action_values_std_order=action["arm_joints"].tolist() + [ action["finger_joints"] ]
        rs_action.update_action(np.multiply(action_values_std_order, self.abs_joint_pos_range.get_values_std_order() ) )
        # Convert action indexing from ur5 to ros
        rs_action = rs_action.joints["ur_j_pos"].get_values_ros_order()

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

        IMPORTANT: gripper should start fully open (max=min=0)
        '''
        self.initial_joint_positions_low = np.array([-0.65, -2.75, 1.0, -3.14, -1.7, -3.14, 0.0])
        self.initial_joint_positions_high = np.array([0.65, -2.0, 2.5, 3.14, -1.0, 3.14, 0.85])

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

        target_pose = self.ur5.get_random_workspace_pose()
        self.rs_state__target_len = len(target_pose)

        return target_pose

    def _get_cube_destination(self):
        pose=np.zeros(6)

        singularity_area = True

        # check if generated x,y,z are in singularityarea
        while singularity_area:
            # Generate random uniform sample in semisphere taking advantage of the
            # sampling rule

            # UR5 workspace radius
            # Max d = 1.892
            R =  0.900 # reduced slightly

            #phi = np.random.default_rng().uniform(low= 0.0, high= 2*np.pi)
            phi = np.random.uniform(low= 0.0, high= 2*np.pi)
            #costheta = np.random.default_rng().uniform(low= 0.0, high= 1.0) # [-1.0,1.0] for a sphere
            costheta = 0
            #u = np.random.default_rng().uniform(low= 0.0, high= 1.0)
            u = np.random.uniform(low= 0.0, high= 1.0)

            theta = np.arccos(costheta)
            r = R * np.cbrt(u)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            if (x**2 + y**2) > 0.085**2:
                singularity_area = False

        pose[:3]=[x, y, z]

        self.rs_state__destination_len = len(pose)
        return pose

    def _get_observation_space(self):
        """Get environment observation space.
        (ur_j_pos + ur_j_vel + gripper_pose)

        Returns:
            gym.spaces: Gym observation space object.

        """

        # Joint position range tolerance
        pos_tolerance = np.full(self.ur5.number_of_joint_positions,0.1)
        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(np.full(self.number_of_joints, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(self.number_of_joints, -1.0), pos_tolerance)
        
        # Joint positions range tolerance
        vel_tolerance = np.full(self.number_of_joints,0.5)
        # Joint velocities range used to determine if there is an error in the sensor readings
        max_joint_velocities = np.add(self.ur5.get_max_joint_velocities().get_values_std_order(), vel_tolerance)
        min_joint_velocities = np.subtract(self.ur5.get_min_joint_velocities().get_values_std_order(), vel_tolerance)

        #gripper pose
        #increase a little bit because 0.85m is the arm length
        max_gripper_pose=[ 1,  1,  1,  np.pi,  np.pi,  np.pi]
        min_gripper_pose=[-1, -1, -1, -np.pi, -np.pi, -np.pi]

        # Definition of environment observation_space
        max_obs = np.concatenate(( max_joint_positions, max_joint_velocities, max_gripper_pose))
        min_obs = np.concatenate(( min_joint_positions, min_joint_velocities, min_gripper_pose))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _get_observation_space_with_cubes(self, number_of_cubes):
        """Get environment observation space, considering the cubes positioning
        ( ur_j_pos + ur_j_vel + gripper_pose + gripper_to_obj_dist + cubes_pose + cubes_destination_pose)

        Returns:
            gym.spaces: Gym observation space object.

        """

        # Joint position range tolerance
        pos_tolerance = np.full(self.ur5.number_of_joint_positions,0.1)
        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(np.full(self.number_of_joints, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(self.number_of_joints, -1.0), pos_tolerance)
        
        # Joint positions range tolerance
        vel_tolerance = np.full(self.number_of_joints,0.5)
        # Joint velocities range used to determine if there is an error in the sensor readings
        max_joint_velocities = np.add(self.ur5.get_max_joint_velocities().get_values_std_order(), vel_tolerance)
        min_joint_velocities = np.subtract(self.ur5.get_min_joint_velocities().get_values_std_order(), vel_tolerance)

        #gripper pose
        #increase a little bit because 0.85m is the arm length and angles in 0.01 because of precision
        angle_tolerance=0.001
        abs_max_angle=np.pi + angle_tolerance #+/-pi precision might fall off space limits

        max_gripper_pose=[ 1,  1,  1,  abs_max_angle,  abs_max_angle,  abs_max_angle]
        min_gripper_pose=[-1, -1, -1, -abs_max_angle, -abs_max_angle, -abs_max_angle]

        #gripper_to_obj_dist
        max_gripper_to_obj_pose=[ 2* 0.9, 2* 0.9, 2* 0.9]#,  abs_max_angle,  abs_max_angle,  abs_max_angle]
        min_gripper_to_obj_pose=[ 2*-0.9, 2*-0.9, 2*-0.9]#, -abs_max_angle, -abs_max_angle, -abs_max_angle]

        #cubes xyzrpy max min
        max_1_cube_pos=[ 0.9,  0.9, np.inf,  abs_max_angle,  abs_max_angle,  abs_max_angle]
        min_1_cube_pos=[-0.9, -0.9,      0, -abs_max_angle, -abs_max_angle, -abs_max_angle]
        max_n_cube_pos=np.array(max_1_cube_pos*number_of_cubes)
        min_n_cube_pos=np.array(min_1_cube_pos*number_of_cubes)
        #cubes destination point xyzrpy max min
        max_cube_destination_pos=[ 0.9,  0.9, np.inf,  abs_max_angle,  abs_max_angle,  abs_max_angle]
        min_cube_destination_pos=[-0.9, -0.9,      0, -abs_max_angle, -abs_max_angle, -abs_max_angle]

        # Definition of environment observation_space
        max_obs = np.concatenate(( max_joint_positions, max_joint_velocities, max_gripper_pose, max_gripper_to_obj_pose, max_n_cube_pos, max_cube_destination_pos))
        min_obs = np.concatenate(( min_joint_positions, min_joint_velocities, min_gripper_pose, min_gripper_to_obj_pose, min_n_cube_pos, min_cube_destination_pos))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _get_action_space(self):
        
        self.action_space = spaces.Dict({
            "arm_joints"    : spaces.Box(low=np.full((self.number_of_arm_joints), -1.0), high=np.full((self.number_of_arm_joints), 1.0), dtype=np.float32),
            "finger_joints" : spaces.Discrete (2) #0-open; 1-close
        })
        
        return self.action_space

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
        rs_state.update_cube_destination(self.cube_destination_pose)

        # Check if the length of the Robot Server state received is correct
        #if not len(rs_state)== self._get_robot_server_state_len():
        #    raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        new_state = rs_state.server_state_to_env_state(robotiq=self.robotiq)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(new_state.to_array() ):
            print(new_state.to_array())
            print(self.observation_space.high)
            print(self.observation_space.low)      
            raise InvalidStateError()

        return new_state, rs_state

    def _get_current_env_state_and_update_observation_space(self):
        """Requests the current robot state (simulated or real), updates obs space according to the number of cubes

        Args:
            NaN

        Returns:
            new_state (env_state): Current state in environment format.
            rs_state (server_state): State in Robot Server format.

        """

        # Get Robot Server state
        rs_state=server_state()
        rs_state.set_server_from_message(np.nan_to_num(np.array(self.client.get_state_msg().state)))
        rs_state.update_cube_destination(self.cube_destination_pose)

        # Check if the length of the Robot Server state received is correct
        #if not len(rs_state)== self._get_robot_server_state_len():
        #    raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        new_state = rs_state.server_state_to_env_state(robotiq=self.robotiq)


        #updates observation space according to the number of cubes
        self.observation_space=self._get_observation_space_with_cubes(new_state.number_of_cubes)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(new_state.to_array() ):
            print(new_state.to_array())
            print(self.observation_space.high)
            print(self.observation_space.low)            
            raise InvalidStateError()

        return new_state, rs_state

class env_state():
    """
    Encapsulates the environment state
    Includes:
        * ur_j_pos (ur_joint_dict)-> robots' joint angles in a ur_joint_dict 
        * ur_j_vel (ur_joint_dict) -> robots' joint velocities in a ur_joint_dict
        * gripper_pose (np.array) -> gripper's pose in xyzrpy
        * cubes_pose (np.array) -> cubes' pose in xyzrpy
        * cubes_destination_pose (np.array) -> cubes' destination pose in xyzrpy

    """
    
    def __init__(self):
        """
        Populates the structure with:
            * ur_j_pos     (ur_joint_dict) : joint positions in angles (with zeros)
            * ur_j_vel     (ur_joint_dict) : joint velocities (with zeros)
            * gripper_pose (np.array) -> gripper's pose in xyzrpy
            * cubes_pose (np.array) -> cubes' pose in xyzrpy
            * cubes_destination_pose (np.array)-> cubes' new pose in xyzrpy
        """
        self.state={
            "ur_j_pos": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ur_j_vel": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "gripper_pose": np.zeros(6, dtype=np.float32),
            "cubes_pose": [],
            "cubes_destination_pose": []
        }
    
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

    def update_cubes_pose(self, new_cubes_pose, number_of_cubes):
        """
        Updates the cubes' pose :
        
        Args:
            new_cubes_pose (np.array): cubes position, new values

        Returns:
            none
        """
        
        self.state["cubes_pose"]=copy.deepcopy(new_cubes_pose)
        self.number_of_cubes=number_of_cubes
    
    def update_gripper_pose(self, new_gripper_pose):
        """
        Updates the gripper pose array
        
        Args:
            new_gripper_pose (array like): new gripper pose info #x y z r p y

        Returns:
            none
        """
        self.state["gripper_pose"]=np.array(new_gripper_pose)

    def update_cube_destination(self, new_cubes_destination):
        """
        Updates the cube destination:
        
        Args:
            new_cubes_destination (array like): new target point in xyzrpy

        Returns:
            none
        """

        self.state["cubes_destination_pose"]=np.array(new_cubes_destination)

    def to_array(self):
        """
        Retrieves the current state as a list. The order is: ( ur_j_pos + ur_j_vel + gripper_pose + gripper_to_obj_dist + cubes_pose + cubes_destination_pose)
        The ur_j_pos and ur_j_vel are displayed in standard order (from base to end effector). Cubes pose ignores index 0 = cube id
        
        Args:
            None

        Returns:
            env_array (list): ordered list containing the current environment's state. The array includes the following: target_polar + ur_j_pos (std order) + ur_j_vel (std_order)+ gripper_pose + cubes_pose + cubes_destination_pose
            for the cubes_pose, the id is ignored [1:]
        """
        gripper_to_obj_pose=self.state["gripper_pose"][0:3] - self.state["cubes_pose"][0, 0:3]

        env_array= self.state["ur_j_pos"].get_values_std_order().tolist() + self.state["ur_j_vel"].get_values_std_order().tolist() + self.state["gripper_pose"].tolist() + gripper_to_obj_pose.tolist() + self.state["cubes_pose"].reshape(-1)[1:].tolist() + self.state["cubes_destination_pose"].tolist()

        return env_array

class action_state():
    """
    Encapsulates an action
    Includes: 
        joints -> dict structure
            * ur_j_pos     (ur_joint_dict) : joint positions in angles (with zeros)
        values -> np arrays, by standard order. Divides arm joints from finger joints to make gripper vs arm controller easier
            * arm_joints (np.array), joints in std order
            * finger_joints (np.array), joints in std order
    """
    
    def __init__(self):
        """
        Populates the structure with:
            joints -> dict structure
                * ur_j_pos     (ur_joint_dict) : joint positions in angles (with zeros)
            values -> np arrays, by standard order. Divides arm joints from finger joints to make gripper vs arm controller easier
                * arm_joints (np.array), joints in std order
                * finger_joints (np.array), joints in std order
        """

        self.joints={
            "ur_j_pos": ur_utils.UR5ROBOTIQ().ur_joint_dict()
        }
        self.values={
            "arm_joints"    : np.array( self.joints["ur_j_pos"].get_arm_joints_value() ),
            "finger_joints" : np.array( self.joints["ur_j_pos"].get_finger_joints_value() )
        }
        self.finger_threshold = 0.01 #open: x<0.01, close:x>=0.01

    def get_action_from_env_state(self, env_state):
        """
        Updates the action joints (joint angles), based on the environment's state
        
        Args:
            env_state (env_state object): current environment's state

        Returns:
            self (So that: new_action=action_state().get_action_from_env_state(env_state) )
        """

        self.update_action(env_state.state["ur_j_pos"].get_values_std_order() )


        return self

    def update_action(self, new_action_std_order):
        """
        Updates the action joints (joint angles), based on array passed in standard order (base to end effector)
        
        Args:
            new_action_std_order (np array): array in std order indicating joint values

        Returns:
            self (So that: new_action=action_state().get_action_from_env_state(env_state) )
        """

        #updates joint dictionary
        self.joints["ur_j_pos"].set_values_std_order(new_action_std_order)

        #finger action either 0-open, 1-close
        for key in self.joints["ur_j_pos"].finger_joints:
            finger_real_value=self.joints["ur_j_pos"].joints[key]
            self.joints["ur_j_pos"].joints[key] = int(0) if self.joints["ur_j_pos"].joints[key] < self.finger_threshold else int(1)

        #updates arm and finger arrays
        self.values ["arm_joints"]    = np.array( self.joints["ur_j_pos"].get_arm_joints_value() )       #std order
        self.values ["finger_joints"] =      int( self.joints["ur_j_pos"].get_finger_joints_value() [0]) #std order


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

        action_array= self.values["arm_joints"].tolist() + self.values["finger_joints"].tolist()

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
        * gripper_pose-> array len=6 x, y, z, r, p, y 
        * cubes_pose-> array len=7 #id, x, y, z, r, p, y
        * cubes_destination_pose-> array len=7 #id, x, y, z, r, p, y

    """
    
    def __init__(self):
        """
        Populates the structure with:
            * target_xyzrpy     (np.array)      : target pose in xyzrpy
            * ur_j_pos          (ur_joint_dict) : joint positions in angles (with zeros)
            * ur_j_vel          (ur_joint_dict) : joint velocities (with zeros)
            * ee_base_transform (np.array)      : end effector base transform
            * collision         (np.array)      : collision array
            * gripper_pose      (np.array)      : where is the gripper in the world frame? #id xyzrpy
            * cubes_pose        (np.array)      : where are the cubes? #id xyzrpy
            * cubes_destination_pose (np.array) : where to move the cubes? xyzrpy
        """
        
        self.state={
            "target_xyzrpy": np.zeros(6, dtype=np.float32), #xyzrpy
            "ur_j_pos": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ur_j_vel": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ee_base_transform": np.zeros(7, dtype=np.float32),
            "collision": np.zeros(1, dtype=np.float32),
            "gripper_pose": np.zeros(6, dtype=np.float32), #xyzrpy
            "cubes_pose": None, #id xyzrpy
            "cubes_destination_pose": None #id xyzrpy
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
    
    def update_gripper_pose(self, new_gripper_pose):
        """
        Updates the gripper pose array
        
        Args:
            new_gripper_pose (array like): new gripper pose info #x y z r p y

        Returns:
            none
        """
        self.state["gripper_pose"]=np.array(new_gripper_pose)

    def update_cubes_pose(self, new_cubes_pose):
        """
        Updates the cubes position
        Each cube is associated with 7 values: #0->id, #1-> x, #2->y, #3-> z, #4->r, #5->p, #6->y
        
        Args:
            new_cubes_pose (array like): cubes positioning

        Returns:
            none
        """
        cubes_info_len=7
        how_many_cubes=int(len(new_cubes_pose)/cubes_info_len)
        self.number_of_cubes=how_many_cubes
        
        self.state["cubes_pose"]=np.zeros((how_many_cubes, cubes_info_len), dtype=np.float32)
        for i in range(how_many_cubes):
            self.state["cubes_pose"][ i, : ]=copy.deepcopy(np.array(new_cubes_pose[i*cubes_info_len:(i+1)*cubes_info_len]))

    def update_cube_destination(self, new_cubes_destination):
        """
        Updates the cube destination:
        
        Args:
            new_cubes_destination (array like): new target point in xyzrpy

        Returns:
            none
        """

        self.state["cubes_destination_pose"]=np.array(new_cubes_destination)  

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
               * gripper pose
               * cubes pose

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
        g= f + len(self.state["gripper_pose"])

        #copies info in the appropriate format
        self.update_target_pose(       msg[ a:b ] )
        self.update_ur_joint_pos(      ur_utils.UR5ROBOTIQ().ur_joint_dict().set_values_ros_order(msg[ b:c ]) )
        self.update_ur_joint_vel(      ur_utils.UR5ROBOTIQ().ur_joint_dict().set_values_ros_order(msg[ c:d ] ))
        self.update_ee_base_transform (msg[ d:e ] )
        self.update_collision(         msg[ e:f ] )
        self.update_gripper_pose(      msg[ f:g ] )
        self.update_cubes_pose(        msg[ g:  ] )
        
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
        #target_coord = np.nan_to_num(self.state["target_xyzrpy"])[0:3]
        #ee_base_transform =   np.nan_to_num(self.state["ee_base_transform"])
        
        #ee_to_base_translation = ee_base_transform[0:3]
        #ee_to_base_quaternion = ee_base_transform[3:8]
        #ee_to_base_rotation = R.from_quat(ee_to_base_quaternion)
        #base_to_ee_rotation = ee_to_base_rotation.inv()
        #base_to_ee_quaternion = base_to_ee_rotation.as_quat()
        #base_to_ee_translation = - ee_to_base_translation

        #target_coord_ee_frame = utils.change_reference_frame(target_coord,base_to_ee_translation,base_to_ee_quaternion)
        #target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)


        ##update
        #new_env_state.update_target_polar(target_polar)
        ur_j_pos_norm=ur_utils.UR5ROBOTIQ(robotiq).normalize_ur_joint_dict(joint_dict=self.state["ur_j_pos"])
        new_env_state.update_ur_j_pos(ur_j_pos_norm)
        new_env_state.update_ur_j_vel(self.state["ur_j_vel"])
        new_env_state.update_gripper_pose(self.state["gripper_pose"])
        new_env_state.update_cubes_pose(self.state["cubes_pose"], self.number_of_cubes)
        new_env_state.update_cube_destination(self.state["cubes_destination_pose"])
        
        return new_env_state

class GraspObjectUR5(UR5RobotiqEnv):
    def _distance_to_goal(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape

        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}

        # Calculate distance to the target
        cubes_destination_pose = np.array(rs_state.state["cubes_destination_pose"])[0:3]
        cube_real_pose         = np.array(rs_state.state["cubes_pose"][ 0, : ])[0:3]  #for now, requests the only cube's pose
        euclidean_dist_3d      = self._distance_to_goal(cubes_destination_pose, cube_real_pose)

        # Reward base
        reward = -1 * euclidean_dist_3d
        
        #Evaluate joint space
        joint_positions_normalized=ur_utils.UR5ROBOTIQ(self.robotiq).normalize_ur_joint_dict(joint_dict=rs_state.state["ur_j_pos"])
        action_values_std_order=action["arm_joints"].tolist() + [ action["finger_joints"] ]
        delta = np.abs(np.subtract(joint_positions_normalized.get_values_std_order(), action_values_std_order ))
        reward = reward - (0.05 * np.sum(delta))

        if euclidean_dist_3d <= self.distance_threshold:
            reward = 100
            done = True
            info['final_status'] = 'success'
            info['cubes_destination_pose'] = cubes_destination_pose

        # Check if robot is in collision
        if rs_state.state["collision"] [-1] == 1:
            collision = True
        else:
            collision = False

        if collision:
            reward = -400
            done = True
            info['final_status'] = 'collision'
            info['cubes_destination_pose'] = cubes_destination_pose

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
            info['cubes_destination_pose'] = cubes_destination_pose

        return reward, done, info


    """        
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

        delta = np.abs(np.subtract(joint_positions_normalized.get_values_std_order(), action.joints["ur_j_pos"].get_values_std_order() ))
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
    """

class GraspObjectUR5Sim(GraspObjectUR5, Simulation):
    #cmd = "roslaunch ur_robot_server ur5Robotiq_sim_robot_server.launch \
    #    max_velocity_scale_factor:=0.2 \
    #    action_cycle_rate:=20"


    cmd = "roslaunch ur_robot_server ur5Robotiq_sim_robot_server.launch \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        world_name:=cubes.world \
        rviz_gui:=false \
        gazebo_gui:=false"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        GraspObjectUR5.__init__(self, rs_address=self.robot_server_ip, robotiq=85, **kwargs)