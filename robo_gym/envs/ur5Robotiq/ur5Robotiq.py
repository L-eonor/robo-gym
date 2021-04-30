#!/usr/bin/env python3

import time
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

class UR5RobotiqEnv(gym.GoalEnv):
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

    def __init__(self, rs_address=None, max_episode_steps=1000, robotiq=85, **kwargs):

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
        #self.abs_joint_pos_range = self.ur5.get_max_joint_positions()
        self.min_joint_pos = np.array(self.ur5.get_min_joint_positions().get_values_std_order())
        self.max_joint_pos = np.array(self.ur5.get_max_joint_positions().get_values_std_order())
        self.initial_joint_positions_low = self.ur_joint_dict()
        self.initial_joint_positions_high = self.ur_joint_dict()

        #simulation params
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
        self.seed()


        # Initialize environment state
        self.state = env_state()
        self.last_position_on_success = []

        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

        #observation space
        self.destination_pose=None
        new_env_state, _ =self._get_current_state() #number of cubes
        self.observation_space = self._get_observation_space_with_cubes(number_of_objs=new_env_state.number_of_cubes)

        #action space
        self.action_space = self._get_action_space()
          
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, initial_joint_positions = None, destination_pose = None, type='random'):
        """Environment reset.

        Args:
            initial_joint_positions (list[7] or np.array[7]): robot joint positions in radians, standard order

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.last_action = None
        self.prev_base_reward = None

        self.destination_pose = [0]*3
        
        ##############################
        # Setting robot server state #
        ##############################

        # Set initial robot joint positions, in standard order
        if initial_joint_positions:
            assert len(initial_joint_positions) == self.ur5.number_of_joint_positions
            ur5_initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            ur5_initial_joint_positions = self.last_position_on_success
        else:
            ur5_initial_joint_positions = self._get_initial_joint_positions()
        #print("initial joint pos")
        #print(ur5_initial_joint_positions)

        # update initial joint positions
        rs_state = server_state()
        rs_state.update_ur_joint_pos(self.ur_joint_dict().set_values_std_order(ur5_initial_joint_positions))
        
        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state = rs_state.get_server_message() )
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")
        
        #############################
        # Reading robot server state#
        #############################

        #Get current state, update obs space with cubes and validate
        self.state, rs_state =self._get_current_state()
        
        # Set destination pose
        if destination_pose:
            assert len(destination_pose) == 6
        else:
            destination_pose = self._get_destination_pose()
        self.destination_pose = destination_pose
        self.state.update_destination_pose(self.destination_pose)
        #print("destination pose after update")
        #print(self.state.state["destination_pose"])


        # check if current position is in the range of the initial joint positions
        if (len(self.last_position_on_success) == 0) or (type=='random'):
            joint_positions=rs_state.get_state()["ur_j_pos"].get_values_std_order()
            tolerance = self.distance_threshold

            for joint in range(len(joint_positions)):
                if (joint_positions[joint]+tolerance < self.initial_joint_positions_low[joint]) or  (joint_positions[joint]-tolerance  > self.initial_joint_positions_high[joint]):
                    print(joint)
                    print(joint_positions[joint])
                    raise InvalidStateError('Reset joint positions are not within defined range')


        # go one empty action and check if there is a collision
        action = self.state.state["ur_j_pos_norm"].get_values_std_order()[0]
        #print("reset action")
        #print(action)

        obs, reward, done, info = self.step( action ) 
        self.elapsed_steps = 0

        if done and info['final_status'] == 'collision':
            print(obs)
            raise InvalidStateError('Reset started in a collision state')

        return obs

    def step(self, action):
        self.elapsed_steps += 1
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Check if the action is within the action space
        assert self.action_space.contains(action ), "%r (%s) invalid" % (action, type(action))

        #create action object and send to robot server
        action_absolute = self._send_action(action)

        # obs, reward, done, info
        obs = self.state.get_obs()
        if not self.observation_space.contains(obs):
            print(obs)
            raise InvalidStateError()

        achieved_goal = np.array(obs['achieved_goal'])#, ndmin=2)
        desired_goal  = np.array(obs['desired_goal'] )#, ndmin=2)

        info, done = self._update_info_and_done(achieved_goal=achieved_goal, desired_goal=desired_goal)

        reward = self.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)            

        return obs, reward, done, info

    def render():
        pass
    
    def _reward(self, rs_state, action):
        return 0, False, {}

    def _send_action(self, action):
        """
        sends action to robot server
        action received is normalized and in std order
        """
        #reformulate action
        rest_of_the_action=self.state.state["ur_j_pos_norm"].get_values_std_order()
        rest_of_the_action[0]=copy.deepcopy(action)
        action=rest_of_the_action
        #print("rest of the action")
        #print(rest_of_the_action)

        # Scale action
        abs_joint_values=np.zeros(len(action), dtype='float32')
        abs_joint_values = (self.max_joint_pos *(1 + action)+ self.min_joint_pos * (1-action))/2
        
        #print("abs joint")
        #print(abs_joint_values)

        #create object to deal with joint order
        new_action_dict=self.ur_joint_dict().set_values_std_order(values=abs_joint_values)

        # Send action to Robot Server
        if not self.client.send_action(new_action_dict.get_values_ros_order().tolist()):
            raise RobotServerError("send_action")  

        self.state, _ = self._get_current_state()

        return abs_joint_values

    #initialization routines

    def _set_initial_joint_positions_range(self):
        '''
        joint positions order: shoulder_pan_joint, shoulder_lift_joint, elbow_joint, writ_1_joint, writ_2_joint, writ_3_joint, finger_joint (0 (open) as initial state)
        (updated the number of joint positions)

        IMPORTANT: gripper should start fully open (max=min=0)
        '''
        #self.initial_joint_positions_low = np.array([-0.65, -2.75, 1.0, -3.14, -1.7, -3.14, 0.0])
        #self.initial_joint_positions_high = np.array([0.65, -2.0, 2.5, 3.14, -1.0, 3.14, 0.85])
        self.initial_joint_positions_low  = np.array(self.ur5.get_min_joint_positions().get_values_std_order())
        self.initial_joint_positions_high = np.array(self.ur5.get_max_joint_positions().get_values_std_order())

    def _get_initial_joint_positions(self):
        """Generate random initial robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """
        self._set_initial_joint_positions_range()
        # Random initial joint positions
        joint_positions = np.random.default_rng().uniform(low=self.initial_joint_positions_low, high=self.initial_joint_positions_high)
        #joint_positions = np.array([-9.36364591e-01, -6.76085472e-01,  9.92242396e-01, -3.15184891e-02, -9.91933823e-01, -9.96397674e-01,  1.00009084e+00], dtype='float32')
        return joint_positions

    def _get_destination_pose(self):
        pose=np.zeros(6)

        #force destination pose to be in the gripper circle
        target_angle = np.random.default_rng().uniform(low=-np.pi, high=np.pi)
        radius=np.linalg.norm(self.state.state["gripper_pose"][0:2])
        x_coordinate=np.cos(target_angle)*radius
        y_coordinate=np.sin(target_angle)*radius
        z_coordinate=copy.deepcopy(self.state.state["gripper_pose"][2])
        pose[0:3]=np.array([x_coordinate, y_coordinate, z_coordinate], dtype='float32')
        return pose
    
    #reward/done/info
    def _update_info_and_done(self, desired_goal, achieved_goal):
        info = {
            'is_success': self._is_success(np.array(achieved_goal), np.array(desired_goal )),
            'final_status': None,
            'destination_pose': self.destination_pose,
        }
        done=False

        euclidean_dist_3d      = self._distance_to_goal(np.array(desired_goal ), np.array(achieved_goal))

        if euclidean_dist_3d.all() <= self.distance_threshold:
            done = True
            info['final_status']='success'
            
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
        
        return info, done

    def _is_success(self, achieved_goal, desired_goal):
        if isinstance(achieved_goal, list):
            achieved_goal = np.array(achieved_goal, dtype='float32')

        if isinstance(desired_goal, list):
            desired_goal = np.array(desired_goal, dtype='float32')
        
        assert achieved_goal.shape == desired_goal.shape
        
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        return (d < self.distance_threshold).astype(np.float32)

    #Observation and action spaces

    def _get_observation_space_with_cubes(self, number_of_objs):
        """Get environment observation space, considering the cubes positioning
        ( ur_j_pos + ur_j_vel + gripper_pose + gripper_to_obj_dist + cubes_pose + destination_pose)

        Returns:
            gym.spaces: Gym observation space object.

        """
        number_of_joints=1
        # Joint position range tolerance
        pos_tolerance = np.full(number_of_joints,self.distance_threshold)
        # Joint positions range used to determine if there is an error in the sensor readings (normalized joints between -1, 1)
        max_joint_positions = np.add(np.full(number_of_joints, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(number_of_joints, -1.0), pos_tolerance)
        
        # Joint positions range tolerance
        vel_tolerance = np.full(number_of_joints,0.5)
        # Joint velocities range used to determine if there is an error in the sensor readings
        max_joint_velocities = np.add(self.ur5.get_max_joint_velocities().get_values_std_order()[0:number_of_joints], vel_tolerance)
        min_joint_velocities = np.subtract(self.ur5.get_min_joint_velocities().get_values_std_order()[0:number_of_joints], vel_tolerance)

        #gripper pose
        #increase a little bit
        # * arm length=0.85m is the arm + gripper attatcher + finger offset + arm above the ground
        # * angles in 0.001 because of precision (pi)
        gripper_tolerance=0.5
        abs_max_gripper_pose=0.85+gripper_tolerance
        angle_tolerance=0.001
        abs_max_angle=np.pi + angle_tolerance #+/-pi precision might fall off space limits
        max_gripper_pose=[ abs_max_gripper_pose,  abs_max_gripper_pose,  abs_max_gripper_pose ]#,  abs_max_angle,  abs_max_angle,  abs_max_angle]
        min_gripper_pose=[-abs_max_gripper_pose, -abs_max_gripper_pose, -abs_max_gripper_pose ]#, -abs_max_angle, -abs_max_angle, -abs_max_angle]

        #gripper_to_obj_dist
        max_gripper_to_obj_pose=[ 2* abs_max_gripper_pose, 2* abs_max_gripper_pose, 2* abs_max_gripper_pose]#,  abs_max_angle,  abs_max_angle,  abs_max_angle]
        min_gripper_to_obj_pose=[ 2*-abs_max_gripper_pose, 2*-abs_max_gripper_pose, 2*-abs_max_gripper_pose]#, -abs_max_angle, -abs_max_angle, -abs_max_angle]

        #cubes xyzrpy (width, depth, height) max min
        max_1_obj_pos=[ abs_max_gripper_pose,  abs_max_gripper_pose, np.inf,  abs_max_angle,  abs_max_angle,  abs_max_angle, np.inf, np.inf, np.inf]
        min_1_obj_pos=[-abs_max_gripper_pose, -abs_max_gripper_pose,      0, -abs_max_angle, -abs_max_angle, -abs_max_angle,      0,      0,      0]
        max_n_obj_pos=np.array(max_1_obj_pos*number_of_objs)
        min_n_obj_pos=np.array(min_1_obj_pos*number_of_objs)

        # Definition of environment observation_space
        max_observation = np.concatenate(( max_joint_positions[0:1], max_joint_velocities[0:1], max_gripper_pose, max_gripper_to_obj_pose, max_n_obj_pos))
        min_observation = np.concatenate(( min_joint_positions[0:1], min_joint_velocities[0:1], min_gripper_pose, min_gripper_to_obj_pose, min_n_obj_pos))


        max_achieved_goal = np.array(max_gripper_pose)
        min_achieved_goal = np.array(min_gripper_pose)

        max_desired_goal = np.array(max_gripper_pose)
        min_desired_goal = np.array(min_gripper_pose)


        self.observation_space = spaces.Dict(dict(
            desired_goal =spaces.Box(low=min_desired_goal,  high=max_desired_goal,  dtype='float32'),
            achieved_goal=spaces.Box(low=min_achieved_goal, high=max_achieved_goal, dtype='float32'),
            observation  =spaces.Box(low=min_observation,   high=max_observation,   dtype='float32'),
        ))
        return self.observation_space
    
    def _get_action_space(self):
        """
        self.action_space = spaces.Dict({
            "arm_joints"    : spaces.Box(low=np.full((self.number_of_arm_joints), -1.0), high=np.full((self.number_of_arm_joints), 1.0), dtype=np.float32),
            "finger_joints" : spaces.Discrete (2) #0-open; 1-close
        })
        """
        #self.action_space = spaces.Box(low=np.full((self.number_of_joints), -1.0), high=np.full((self.number_of_joints), 1.0), dtype=np.float32)
        self.action_space = spaces.Box(low=np.full(1, -1.0), high=np.full(1, 1.0), dtype=np.float32)
        
        return self.action_space

    #get state

    def _get_current_state(self):
        """Requests the current robot state (simulated or real)

        Args:
            NaN

        Returns:
            new_state (env_state): Current state in environment format.
            rs_state (server_state): State in Robot Server format.

        """
        # Get Robot Server state
        rs_state=server_state()
        rs_state.set_server_from_message(np.nan_to_num(np.array(self.client.get_state_msg().state)), self.destination_pose)

        # Convert the initial state from Robot Server format to environment format
        new_state = rs_state.server_state_to_env_state(robotiq=self.robotiq)

        return new_state, rs_state

class env_state():
    """
    Encapsulates the environment state
    Includes:
        * ur_j_pos (ur_joint_dict)-> robots' joint angles in a ur_joint_dict 
        * ur_j_vel (ur_joint_dict) -> robots' joint velocities in a ur_joint_dict
        * gripper_pose (np.array) -> gripper's pose in xyzrpy
        * cubes_pose (np.array) -> cubes' pose in #id, x, y, z, r, p, y,width,->depth, -height
        * destination_pose (np.array) -> cubes' destination pose in xyzrpy
        * collision-> is the robot is collision?

    """
    
    def __init__(self):
        """
        Populates the structure with:
            * ur_j_pos     (ur_joint_dict) : joint positions in angles (with zeros)
            * ur_j_pos_norm     (ur_joint_dict) : same as previous, but the joints are normalized between -1, 1
            * ur_j_vel     (ur_joint_dict) : joint velocities (with zeros)
            * gripper_pose (np.array) -> gripper's pose in xyzrpy
            * cubes_pose (np.array) -> cubes' pose in xyzrpy
            * destination_pose (np.array)-> cubes' new pose in xyzrpy
        """
        self.state={
            "ur_j_pos": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ur_j_pos_norm": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ur_j_vel": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "gripper_pose": np.zeros(6, dtype=np.float32),
            "cubes_pose": [],
            "destination_pose": [],
            "collision":[]
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
        self.state["ur_j_pos_norm"]=ur_utils.UR5ROBOTIQ().normalize_ur_joint_dict(joint_dict=ur_joint_state)

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

    def update_destination_pose(self, new_cubes_destination):
        """
        Updates the cube destination:
        
        Args:
            new_cubes_destination (array like): new target point in xyzrpy

        Returns:
            none
        """

        self.state["destination_pose"]=np.array(new_cubes_destination)

    def update_collision (self, new_collision_state):
        """
        Updates the cllision state:
        
        Args:
            new_collision_state (array like): is the robot in collision state?

        Returns:
            none
        """

        self.state["collision"]=np.array(new_collision_state)

    def _get_target_to_gripper(self):
        """
        Returns the object position in relation to gripper
        """

        return self.state["destination_pose"][0:3] - self.state["gripper_pose"][0:3]

    def to_array(self):
        """
        Retrieves the current state as a list. The order is: ( ur_j_pos + ur_j_vel + gripper_pose + gripper_to_obj_dist + cubes_pose + destination_pose)
        The ur_j_pos and ur_j_vel are displayed in standard order (from base to end effector). Cubes pose ignores index 0 = cube id
        
        Args:
            None

        Returns:
            env_array (list): ordered list containing the current environment's state. The array includes the following: target_polar + ur_j_pos (std order) + ur_j_vel (std_order)+ gripper_pose + cubes_pose + destination_pose
            for the cubes_pose, the id is ignored [1:]
        """
        gripper_to_obj_pose = self._get_target_to_gripper()

        #env_array= self.state["ur_j_pos"].get_values_std_order().tolist() + self.state["ur_j_vel"].get_values_std_order().tolist() + self.state["gripper_pose"].tolist() + gripper_to_obj_pose.tolist() + self.state["cubes_pose"][:, 1:].reshape(-1).tolist() + self.state["destination_pose"].tolist()
        env_array= self.state["ur_j_pos"].get_values_std_order().tolist() + self.state["ur_j_vel"].get_values_std_order().tolist() + self.state["gripper_pose"].tolist() + gripper_to_obj_pose.tolist() + self.state["cubes_pose"][:, 1:].reshape(-1).tolist() + self.state["destination_pose"].tolist()
        print("env array")
        print(env_array)
        return env_array

    def get_obs(self):
        gripper_to_obj_pose = self._get_target_to_gripper()
        
        obs=dict(
            #where to put the gripper? in the cube to reach
            desired_goal = self.state["destination_pose"][0:3].reshape(-1) ,
            #where the gripper really is
            achieved_goal= self.state["gripper_pose"][0:3].reshape(-1) ,
            observation  = np.concatenate([self.state["ur_j_pos_norm"].get_values_std_order()[0:1], self.state["ur_j_vel"].get_values_std_order()[0:1], self.state["gripper_pose"][0:3], gripper_to_obj_pose, self.state["cubes_pose"][:, 1:].reshape(-1)]).reshape(-1)
        )

        return obs

    def get_obj_limits(self):
        """
        returns the limit points of each object
        """
        limit_points=np.zeros((self.number_of_cubes, 8, 3))
        index=0
        for obj in self.state["cubes_pose"]:
            #gets depth, width, height
            depth = obj[-3]
            width = obj[-2]
            height= obj[-1]

            #gets roll, pitch, yaw
            roll = obj[6] #roll->x
            pitch= obj[5]
            yaw  = obj[4] #(yaw, the rotation at x, from the gazebo means z

            #central point
            center_of_mass = np.array(obj[1:4])

            #computes x, y, z abs vaolues for each point
            x_abs = depth/2
            y_abs = width/2
            z_abs = height/2

            #roll, pitch yaw rotation matrix
            R_z_roll =[np.cos(roll) , -np.sin(roll), 0            , np.sin(roll), np.cos(roll), 0           , 0             , 0          , 1            ]
            R_y_pitch=[np.cos(pitch), 0            , np.sin(pitch), 0           , 1           , 0           , -np.sin(pitch), 0          , np.cos(pitch)]
            R_x_yaw  =[1            , 0            , 0            , 0           , np.cos(yaw) , -np.sin(yaw), 0             , np.sin(yaw), np.cos(yaw)  ]

            R_z_roll =np.reshape(R_z_roll,  (3, 3))
            R_y_pitch=np.reshape(R_y_pitch, (3, 3))
            R_x_yaw  =np.reshape(R_x_yaw,   (3, 3))

            R=np.matmul(R_z_roll, R_y_pitch)
            R=np.matmul(R, R_x_yaw)

            #limit points with rotation
            A = center_of_mass + R.dot(np.array([+ x_abs, - y_abs, - z_abs]))
            B = center_of_mass + R.dot(np.array([+ x_abs, + y_abs, - z_abs]))
            C = center_of_mass + R.dot(np.array([- x_abs, - y_abs, - z_abs]))
            D = center_of_mass + R.dot(np.array([- x_abs, + y_abs, - z_abs]))
            E = center_of_mass + R.dot(np.array([+ x_abs, - y_abs, + z_abs]))
            F = center_of_mass + R.dot(np.array([+ x_abs, + y_abs, + z_abs]))
            G = center_of_mass + R.dot(np.array([- x_abs, - y_abs, + z_abs]))
            H = center_of_mass + R.dot(np.array([- x_abs, + y_abs, + z_abs]))

            limit_points_1_obj=np.reshape([A, B, C, D, E, F, G, H],(8, 3))
            limit_points[index, :, :]=copy.deepcopy(limit_points_1_obj)
            index +=1

        return limit_points

class action_state():
    """
    Encapsulates an action
    Includes: 
        joints -> dict structure
            * ur_j_pos_norm     (ur_joint_dict) : joint positions in angles (with zeros)
        values -> np arrays, by standard order. Divides arm joints from finger joints to make gripper vs arm controller easier
            * arm_joints (np.array), joints in std order
            * finger_joints (np.array), joints in std order
    """
    
    def __init__(self):
        """
        Populates the structure with:
            joints -> dict structure
                * ur_j_pos_norm     (ur_joint_dict) : joint positions in angles (with zeros)
            values -> np arrays, by standard order. Divides arm joints from finger joints to make gripper vs arm controller easier
                * arm_joints (np.array), joints in std order
                * finger_joints (np.array), joints in std order
        """

        self.joints={
            "ur_j_pos_norm": ur_utils.UR5ROBOTIQ().ur_joint_dict()
        }
        self.values={
            "arm_joints"    : np.array( self.joints["ur_j_pos_norm"].get_arm_joints_value() ),
            "finger_joints" : np.array( self.joints["ur_j_pos_norm"].get_finger_joints_value() )
        }
        self.finger_threshold = 0.01 #open: x<0.01, close:x>=0.01

    def update_action(self, new_action_std_order):
        """
        Updates the action joints (joint angles), based on array passed in standard order (base to end effector)
        finger action either 0-open, 1-close
        
        Args:
            new_action_std_order (np array): array in std order indicating joint values

        Returns:
            self (So that: new_action=action_state().update_action(env_state) )
        """

        #updates joint dictionary
        self.joints["ur_j_pos_norm"].set_values_std_order(new_action_std_order)

        #finger action either 0-open, 1-close
        for key in self.joints["ur_j_pos_norm"].finger_joints:
            self.joints["ur_j_pos_norm"].joints[key] = int(0) if self.joints["ur_j_pos_norm"].joints[key] < self.finger_threshold else int(1)

        #updates arm and finger arrays
        self.values ["arm_joints"]    = np.array( self.joints["ur_j_pos_norm"].get_arm_joints_value() )       #std order
        self.values ["finger_joints"] =      int( self.joints["ur_j_pos_norm"].get_finger_joints_value() [0]) #std order


        return self

    def action_as_box(self):
        """
        Updates the action joints (joint angles), based on array passed in standard order (base to end effector)
        finger action either 0-open, 1-close
        
        Args:
            new_action_std_order (np array): array in std order indicating joint values

        Returns:
        """
        arm_joints=copy.deepcopy(self.values ["arm_joints"])
        finger_joint = copy.deepcopy(self.values ["finger_joints"])

        #converts from box (required to run baselines) to driver encoding 0-> open , 1->close
        finger_joint= float(-1) if finger_joint < self.finger_threshold else int(1)

        return arm_joints.tolist() + [finger_joint]
        
    def to_array(self):
        """
        Retrieves the action as a list. The order is: (ur_j_pos_norm  )
        The ur_j_pos_norm  are displayed in standard order (from base to end effector)
        
        Args:
            None

        Returns:
            action_array (list): ordered list containing the current action description. The array includes the following: ur_j_pos_norm (std order)
        """

        action_array= self.values["arm_joints"].tolist() + self.values["finger_joints"].tolist()

        return action_array

class server_state():
    """
    Encapsulates the robot server state
    Includes:
        * ur_j_pos-> robots' joint angles in a ur_joint_dict
        * ur_j_vel-> robots' joint velocities in a ur_joint_dict
        * collision-> array len=1
        * gripper_pose-> array len=6 x, y, z, r, p, y 
        * cubes_pose-> array len=7 #id, x, y, z, r, p, y,width,->depth, -height

    """
    
    def __init__(self):
        """
        Populates the structure with:
            * ur_j_pos          (ur_joint_dict) : joint positions in angles (with zeros)
            * ur_j_vel          (ur_joint_dict) : joint velocities (with zeros)
            * collision         (np.array)      : collision array
            * gripper_pose      (np.array)      : where is the gripper in the world frame? #id xyzrpy
            * cubes_pose        (np.array)      : where are the cubes? #id xyzrpy
            * destination_pose (np.array) : where to move the cubes? xyzrpy
        """
        
        self.state={
            "ur_j_pos": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ur_j_vel": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "collision": np.zeros(1, dtype=np.float32),
            "gripper_pose": np.zeros(6, dtype=np.float32), #xyzrpy
            "cubes_pose": None, #id xyzrpy
            "destination_pose": None #id xyzrpy
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

        msg= self.state["ur_j_pos"].get_values_ros_order().tolist() + self.state["ur_j_vel"].get_values_ros_order().tolist() + self.state["collision"].tolist()
        
        return msg

    def update_ur_joint_pos(self, new_joint_pos):
        """
        Updates the joints' positions in angles and normalizes joints between -pi and pi
        
        Args:
            new_joint_pos (ur_joint_dict): Joint position object, new values

        Returns:
            none
        """
        joints=new_joint_pos.get_values_std_order()
        for i in range(len(joints)):
            while joints[i] > np.pi:
                joints[i]-=np.pi
            while joints[i] < -np.pi:
                joints[i]+=np.pi
        self.state["ur_j_pos"]=new_joint_pos.set_values_std_order(joints)

    def update_ur_joint_vel(self, new_joint_vel):
        """
        Updates the joints' velocities:
        
        Args:
            new_joint_pos (ur_joint_dict): Joint vel object, new values

        Returns:
            none
        """

        self.state["ur_j_vel"]=copy.deepcopy(new_joint_vel)
    
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
        Each cube is associated with 10 values: #0->id, #1-> x, #2->y, #3-> z, #4->r, #5->p, #6->y, #7->width #8->depth, #9-height
        
        Args:
            new_cubes_pose (array like): cubes positioning

        Returns:
            none
        """
        cubes_info_len=10
        how_many_cubes=int(len(new_cubes_pose)/cubes_info_len)
        self.number_of_cubes=how_many_cubes

        #change order: from id, (w, d, h), (x, y, z), (w, d, h) TO id, x, y, z, r, p, y, w, d, h
        new_cubes_pose=np.reshape(new_cubes_pose, (-1, cubes_info_len))

        self.state["cubes_pose"]=copy.deepcopy(np.hstack((new_cubes_pose[:, 0:1], new_cubes_pose[:, 4:], new_cubes_pose[:, 1:4])))

    def update_destination_pose(self, new_cubes_destination):
        """
        Updates the cube destination:
        
        Args:
            new_cubes_destination (array like): new target point in xyzrpy

        Returns:
            none
        """

        self.state["destination_pose"]=np.array(new_cubes_destination)  

    def set_server_from_message(self, msg, destination_pose):
        """
        Updates the state values: position, velocity and collision. Uses the info retrieved from the server by the corresponding message array
        
        Args:
            msg (list): server's state to be saved. Includes
               * joint's position (angles) in ros order (alphabetical)-> elbow_joint, finger_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint
               * joint's velocities in ros order (alphabetical)-> elbow_joint, finger_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint
               * collision info
               * gripper pose
               * cubes pose

        Returns:
            None
        """

        #computes the list indexes here to make the code easily readable
        a= 0
        b= a + len(self.state["ur_j_pos"].joints)
        c= b + len(self.state["ur_j_vel"].joints)
        d= c + len(self.state["collision"])
        e= d + len(self.state["gripper_pose"])

        #copies info in the appropriate format
        self.update_ur_joint_pos(      ur_utils.UR5ROBOTIQ().ur_joint_dict().set_values_ros_order(msg[ a:b ]) )
        self.update_ur_joint_vel(      ur_utils.UR5ROBOTIQ().ur_joint_dict().set_values_ros_order(msg[ b:c ] ))
        self.update_collision(         msg[ c:d ] )
        self.update_gripper_pose(      msg[ d:e ] )
        self.update_cubes_pose(        msg[ e:  ] )
        self.update_destination_pose(destination_pose)
        
    def server_state_to_env_state(self, robotiq=85):
        """
        Creates the environment's state object based on the server state object. This means updating the environment state
        
        Args:
            robotiq (int): (85 or 140) reference to gripper model, in order to choose the appropriate joint limits

        Returns:
            new_env_state (env_state) -> new env state object with updated values
        """

        new_env_state=env_state()

        ##update
        new_env_state.update_ur_j_pos(self.state["ur_j_pos"])
        new_env_state.update_ur_j_vel(self.state["ur_j_vel"])
        new_env_state.update_gripper_pose(self.state["gripper_pose"])
        #consider all cubes
        new_env_state.update_cubes_pose(self.state["cubes_pose"], self.number_of_cubes)
        new_env_state.update_destination_pose(self.state["destination_pose"])
        new_env_state.update_collision(self.state["collision"])
        
        return new_env_state

class GraspObjectUR5(UR5RobotiqEnv):
    def _distance_to_goal(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape

        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = 0
        done = False
        
        # Calculate distance to the target
        #desired goal
        destination_pose = desired_goal#np.array(desired_goal )
        #achieved goal
        cube_real_pose         = achieved_goal#np.array(achieved_goal)  #for now, requests the only cube's pose
        #euclidean norm
        euclidean_dist_3d      = self._distance_to_goal(destination_pose, cube_real_pose).reshape(-1)

        # Reward base
        reward = -1 * euclidean_dist_3d
        
        corrected_reward=np.array([100.0 if np.absolute(r)<=self.distance_threshold else r for r in reward], dtype='float32')

        if len(corrected_reward)==1:
            corrected_reward=corrected_reward[0]

        return corrected_reward

class GraspObjectUR5Sim(GraspObjectUR5, Simulation):
    #cmd = "roslaunch ur_robot_server ur5Robotiq_sim_robot_server.launch \
    #    max_velocity_scale_factor:=0.2 \
    #    action_cycle_rate:=20"


    cmd = "roslaunch ur_robot_server ur5Robotiq_sim_robot_server.launch \
        max_velocity_scale_factor:=0.6 \
        action_cycle_rate:=20 \
        world_name:=cubes.world \
        rviz_gui:=false \
        gazebo_gui:=true"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        GraspObjectUR5.__init__(self, rs_address=self.robot_server_ip, max_episode_steps=1000, robotiq=85, **kwargs)