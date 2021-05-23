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
        self.min_joint_pos = np.array(self.ur5.get_min_joint_positions().get_values_std_order())
        self.max_joint_pos = np.array(self.ur5.get_max_joint_positions().get_values_std_order())
        self.initial_joint_positions_low = self.ur_joint_dict()
        self.initial_joint_positions_high = self.ur_joint_dict()

        self.distance_threshold = 0.02 #distance to cube, to be considered well positioned
        self.finger_threshold = 0.1 #open: x<0.01, close:x>=0.01
        self.gripper_error_threshold=0.05
        #self.grasp_threshold=0.03  #distance required between the gripper and the object to perform grasping

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
        new_env_state, _ =self._get_current_state() #this is required to get the number of cubes
        self.observation_space = self._get_observation_space_with_cubes(number_of_objs=new_env_state.number_of_cubes)

        #action space
        self.action_space = self._get_action_space()

        #kinematics:
        self.kinematics=ur_utils.kinematics_model(ur_model='ur5')
          
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
        '''
        Isto tem de ser mudado para ser desired pose + gripper state (como na action)
        '''
        # Set initial robot joint positions, in standard order
        if False:#initial_joint_positions:
            assert len(initial_joint_positions) == self.ur5.number_of_joint_positions
            ur5_initial_joint_positions = initial_joint_positions
        elif False:#(len(self.last_position_on_success) != 0) and (type=='continue'):
            ur5_initial_joint_positions = self.last_position_on_success
        else:
            ur5_initial_joint_positions = self._get_initial_joint_positions()        
        reset_pose, new_ee_orientation=self.kinematics.forward_kin(ur5_initial_joint_positions[0:6])

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

        #verifies if the gripper was correctly reseted
        if np.absolute(np.linalg.norm(self.state.state["gripper_pose_gazebo"][0:3] - reset_pose, axis=-1)) > self.gripper_error_threshold:
            raise RobotServerError("gripper")
        
        '''
        Isto tem de ser mudado para o goal 
        '''
        # Set destination pose
        if destination_pose:
            assert len(destination_pose) == 3
        else:
            destination_pose = self._get_destination_pose()
        self.destination_pose = destination_pose
        self.state.update_destination_pose(self.destination_pose)

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
        #action = np.concatenate((self.state.state["gripper_pose"], [0], [0]))
        action = np.concatenate((self.state.state["gripper_pose"], [0]))
        
        obs, reward, done, info = self.step( action ) 
        self.elapsed_steps = 0

        return obs

    def step(self, action):
        self.elapsed_steps += 1
        
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Check if the action is within the action space
        assert self.action_space.contains(action ), "%r (%s) invalid" % (action, type(action))

        #create action object and send to robot server
        joints_absolute = self._send_action(action=action)

        # obs, reward, done, info
        obs = self.state.get_obs()
        if not self.observation_space.contains(obs):
            print(obs)
            raise InvalidStateError()


        #verifies if the gripper was correctly reseted
        if np.absolute(np.linalg.norm(self.state.state["gripper_pose"] - self.state.state["gripper_pose_gazebo"][0:3], axis=-1)) > self.gripper_error_threshold:
            raise RobotServerError("gripper")

        #achieved_goal = np.array(self.state.state["cubes_pose"][0, 1:4].reshape(-1) )
        #desired_goal  = np.array(self.state.state["destination_pose"][0:3].reshape(-1) )
        desired_goal= np.array(self.state.state["cubes_pose"][0, 1:4].reshape(-1) )
        achieved_goal= self.state.state["gripper_pose"]

        info, done = self._update_info_and_done(achieved_goal=achieved_goal, desired_goal=desired_goal)

        if joints_absolute is not None:
            reward = self.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info) * 1.0    
        else:
            reward = -100.0
        #return obs['observation'], reward, done, info['destination_pose']
        return obs, reward, done, info

    def render():
        pass
    
    def _reward(self, rs_state, action):
        print("WRONG FUNCTION")
        return 0, False, {}

    def _send_action(self, action, gripper=0):
        
        #pose
        object_pose=action[0:3]
        gripper_dest_orientation=0#action[3]
        gripper_state=0#action[3]#action[4]

        #if no cube grasped, required to be perpendicular to the closest cube
        #if gripper is near obj, mantain the orientation perpendicular to obj
        if not self._is_grasping():
            gripper_orientation=self._get_grasp_orientation()
        #if grasping, orientation is 0 or pi/2
        else:
            #gripper_orientation=float(0) if action[3] <np.pi/4 else float (np.pi/2)
            gripper_orientation=gripper_dest_orientation

        gripper=int(0) if gripper_state < self.finger_threshold else int(1)


        #inverse kinematics
        #ee orientation
        ee_orientation=np.array([[np.cos(gripper_orientation), np.sin(gripper_orientation), 0], [np.sin(gripper_orientation), -np.cos(gripper_orientation), 0], [0, 0, -1]])
        #current robot joints
        current_joints=self.state.state["ur_j_pos"].get_arm_joints_value()
        desired_joints_std_order=self.kinematics.get_joint_combination(pose=object_pose, orientation=ee_orientation, current_joints=current_joints)
        if desired_joints_std_order is not None:
            desired_joints_std_order_with_gripper=np.concatenate((desired_joints_std_order, [gripper]))        

            #create object to deal with joint order
            new_action_dict=self.ur_joint_dict().set_values_std_order(values=desired_joints_std_order_with_gripper)

            # Send action to Robot Server
            if not self.client.send_action(new_action_dict.get_values_ros_order().tolist()):
                raise RobotServerError("send_action")  

            self.state, _ = self._get_current_state()

            return desired_joints_std_order
        else:
            return None

    #initialization routines

    def _set_initial_joint_positions_range(self):
        '''
        joint positions order: shoulder_pan_joint, shoulder_lift_joint, elbow_joint, writ_1_joint, writ_2_joint, writ_3_joint, finger_joint (0 (open) as initial state)
        (updated the number of joint positions)

        IMPORTANT: gripper should start fully open (max=min=0)
        '''
        self.initial_joint_positions_low  = np.array(self.ur5.get_min_joint_positions().get_values_std_order())
        self.initial_joint_positions_high = np.array(self.ur5.get_max_joint_positions().get_values_std_order())

    def _get_initial_joint_positions(self):
        """Generate random initial robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """
        self._set_initial_joint_positions_range()
        # Random initial joint positions
        #joint_positions = np.random.default_rng().uniform(low=self.initial_joint_positions_low, high=self.initial_joint_positions_high)
        joint_positions=[0, -1.225197, 1.1146594, -1.4602588, -1.5707965, -1.5853374, 0 ]
        return joint_positions

    def _get_destination_pose(self):
        cube_height=self.state.state["cubes_pose"][0, 9]
        #force destination pose to be in [0.5, 0.5, height/2]
        pose=[0.25, 0.25, cube_height/2]

        return pose
    
    #reward/done/info
    def _update_info_and_done(self, desired_goal, achieved_goal):
        euclidean_dist_3d      = np.absolute(self._distance_to_goal(desired_goal, achieved_goal))
        done= euclidean_dist_3d<=self.distance_threshold
        info = {
            'is_success': done,
            'final_status': 'sucess' if done else 'Not final status',
            'destination_pose': self.destination_pose,
        }
            
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
        
        return info, done

    #Observation and action spaces
    def _get_observation_space_with_cubes(self, number_of_objs):
        """Get environment observation space, considering the cubes positioning
        ( ur_j_pos + ur_j_vel + gripper_pose_gazebo + gripper_to_obj_dist + cubes_pose + destination_pose)

        Returns:
            gym.spaces: Gym observation space object.

        """
        '''
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
        '''
        #gripper pose
        #increase a little bit
        # * arm length=0.85m is the arm + gripper attatcher + finger offset + arm above the ground
        # * angles in 0.001 because of precision (pi)
        gripper_tolerance=0.5
        abs_max_gripper_pose=0.85+gripper_tolerance
        angle_tolerance=0.001
        abs_max_angle=np.pi + angle_tolerance #+/-pi precision might fall off space limits
        max_gripper_pose=[ abs_max_gripper_pose,  abs_max_gripper_pose,  abs_max_gripper_pose ]#,  abs_max_angle,  abs_max_angle,  abs_max_angle]
        min_gripper_pose=[-0.1, -abs_max_gripper_pose, -0.1 ]#, -abs_max_angle, -abs_max_angle, -abs_max_angle]

        #gripper_to_obj_dist
        max_gripper_to_obj_pose=[ 2* abs_max_gripper_pose, 2* abs_max_gripper_pose, 2* abs_max_gripper_pose]#,  abs_max_angle,  abs_max_angle,  abs_max_angle]
        min_gripper_to_obj_pose=[ 1*-abs_max_gripper_pose, 2*-abs_max_gripper_pose, 2*-abs_max_gripper_pose]#, -abs_max_angle, -abs_max_angle, -abs_max_angle]

        #cubes xyzrpy (width, depth, height) max min
        #max_1_obj_pos=[ abs_max_gripper_pose,  abs_max_gripper_pose, np.inf,  abs_max_angle,  abs_max_angle,  abs_max_angle, np.inf, np.inf, np.inf]
        #min_1_obj_pos=[-abs_max_gripper_pose, -abs_max_gripper_pose,      0, -abs_max_angle, -abs_max_angle, -abs_max_angle,      0,      0,      0]
        #cubes xyz, roll, height
        max_1_obj_pos=[ abs_max_gripper_pose,  abs_max_gripper_pose, np.inf]#,  abs_max_angle, np.inf]
        min_1_obj_pos=[0, -abs_max_gripper_pose,      0]#, -abs_max_angle,      -0.1]#0]
        max_n_obj_pos=np.array(max_1_obj_pos*number_of_objs)
        min_n_obj_pos=np.array(min_1_obj_pos*number_of_objs)

        # Definition of environment observation_space
        max_observation = np.concatenate(( max_gripper_pose, max_gripper_to_obj_pose, max_n_obj_pos))
        min_observation = np.concatenate(( min_gripper_pose, min_gripper_to_obj_pose, min_n_obj_pos))

        number_of_goals=1
        max_achieved_goal = np.array(max_gripper_pose)
        min_achieved_goal = np.array(min_gripper_pose)

        max_desired_goal = np.array(max_gripper_pose)
        min_desired_goal = np.array(min_gripper_pose)


        self.observation_space = spaces.Dict(dict(
            desired_goal =spaces.Box(low=min_desired_goal,  high=max_desired_goal,  dtype='float32'),
            achieved_goal=spaces.Box(low=min_achieved_goal, high=max_achieved_goal, dtype='float32'),
            observation  =spaces.Box(low=min_observation,   high=max_observation,   dtype='float32'),
        ))
        self.observation_space=spaces.Box(low=min_observation,   high=max_observation,   dtype='float32')

        return self.observation_space
    
    def _get_action_space(self):
        """
        self.action_space = spaces.Dict({
            "arm_joints"    : spaces.Box(low=np.full((self.number_of_arm_joints), -1.0), high=np.full((self.number_of_arm_joints), 1.0), dtype=np.float32),
            "finger_joints" : spaces.Discrete (2) #0-open; 1-close
        })
        """
        #action is the gripper pose and open/close
        #gripper pose
        #increase a little bit
        # * arm length=0.85m is the arm + gripper attatcher + finger offset + arm above the ground
        # * angles in 0.001 because of precision (pi)
        gripper_tolerance=0.5
        abs_max_gripper_pose=0.85+gripper_tolerance
        angle_tolerance=0.001
        abs_max_angle=np.pi + angle_tolerance #+/-pi precision might fall off space limits
        #the gripper's orientation is fixed pointed down
        #max_gripper_pose=np.array([ abs_max_gripper_pose,  abs_max_gripper_pose,  0.08])#,  abs_max_angle,  abs_max_angle,  abs_max_angle]
        #min_gripper_pose=np.array([-abs_max_gripper_pose, -abs_max_gripper_pose, 0.02])#, -abs_max_angle, -abs_max_angle, -abs_max_angle]
        max_gripper_pose=np.array([ 0.60,   0.2, 0.08])#,  abs_max_angle,  abs_max_angle,  abs_max_angle]
        min_gripper_pose=np.array([ 0.20 , -0.2, 0.01])#, -abs_max_angle, -abs_max_angle, -abs_max_angle]

        max_gripper_angle=[np.pi/2]
        min_gripper_angle=[0]

        max_gripper_state=[1]
        min_gripper_state=[0]
        
        #action_max=np.concatenate(( max_gripper_pose, max_gripper_state))
        #action_min=np.concatenate(( min_gripper_pose, min_gripper_state))
        action_max=max_gripper_pose
        action_min=min_gripper_pose

        self.action_space = spaces.Box(low=action_min, high=action_max, dtype=np.float32)

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

    #grasp functions
    def _is_grasping(self):
        #open: x<0.01, close:x>=0.01
        is_gripper_closed=False if self.state.state["ur_j_pos_norm"].get_finger_joints_value() < -0.5 else True
        is_gripper_near_obj=self._gripper_is_near_obj()
        is_grasping=is_gripper_near_obj and is_gripper_closed

        return is_grasping

    def _gripper_is_near_obj(self):
        #gripper_pose =self.state.state['gripper_pose_gazebo'][0:3]
        gripper_pose =self.state.state['gripper_pose']
        
        closest_object=self._get_closest_obj()
        obj_pose = closest_object[1:4]
        self.grasp_threshold=closest_object[9] + 0.05 #threshold distance between the cube and the gripper must be the >= as the height
        gripper_to_obj=np.linalg.norm(gripper_pose - obj_pose, axis=-1)

        return (gripper_to_obj <= self.grasp_threshold)

    def _get_closest_obj(self):

        closest_object_index=np.argmin(np.linalg.norm(self.state.state["cubes_pose"][:, 1:4]-self.state.state["gripper_pose"], axis=1), axis=0)
        closest_object=self.state.state["cubes_pose"][closest_object_index, :]

        return closest_object

    def _get_grasp_orientation(self):
        closest_object=self._get_closest_obj()
        obj_orientation=closest_object[6]

        #restricts object orientation to interval 0-pi, which is the meaningful range to define gripper pose
        while obj_orientation>np.pi:
            obj_orientation-=np.pi
        while obj_orientation<0:
            obj_orientation+=np.pi
        #gripper perpendicular to obj
        if obj_orientation<np.pi/2:
            grasp_orientation=obj_orientation+np.pi/2
        else:
            grasp_orientation=obj_orientation-np.pi/2
                   
        return grasp_orientation

class env_state():
    """
    Encapsulates the environment state
    Includes:
        * gripper_pose (np array)-> gripper pose from inverse kinematics based on the joint values (x, y, z)
        * gripper_pose_gazebo (np.array) -> gripper's pose in xyzrpy, estimated by the gazebo enginex, y, z in the base frame
        * cubes_pose (np.array) -> cubes' pose in #id, x, y, z, r, p, y,width,->depth, -height
        * destination_pose (np.array) -> cubes' destination pose in xyzrpy
        * ur_j_pos (ur_joint_dict)-> robots' joint angles in a ur_joint_dict 
        * ur_j_vel (ur_joint_dict) -> robots' joint velocities in a ur_joint_dict
        * collision-> is the robot is collision?

    """
    
    def __init__(self):
        """
        Populates the structure with:
            * gripper_pose (np array)-> gripper pose from inverse kinematics based on the joint values (x, y, z)
            * gripper_pose_gazebo (np.array) -> gripper's pose in xyzrpy, estimated by the gazebo enginex, y, z in the base frame
            * cubes_pose (np.array) -> cubes' pose in #id, x, y, z, r, p, y,width,->depth, -height
            * destination_pose (np.array) -> cubes' destination pose in xyzrpy

            * ur_j_pos     (ur_joint_dict) : joint positions in angles (with zeros)
            * ur_j_pos_norm     (ur_joint_dict) : same as previous, but the joints are normalized between -1, 1
            * ur_j_vel     (ur_joint_dict) : joint velocities (with zeros)
        """
        self.state={
            "gripper_pose": np.zeros(3, dtype=np.float32),
            "ur_j_pos": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ur_j_pos_norm": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ur_j_vel": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "gripper_pose_gazebo": np.zeros(6, dtype=np.float32),
            "cubes_pose": [],
            "destination_pose": [],
            "collision":[]
        }

        self.kinematics=ur_utils.kinematics_model(ur_model='ur5')  
    
    def update_ur_j_pos(self, ur_joint_state):
        """
        Updates the joints' positions in angles and gripper pose computed through kinematics:
        
        Args:
            ur_joint_state (ur_joint_dict): Joint position object, new values

        Returns:
            none
        """
        
        #joints true value
        self.state["ur_j_pos"]=copy.deepcopy(ur_joint_state)
        #joints normalized values
        self.state["ur_j_pos_norm"]=ur_utils.UR5ROBOTIQ().normalize_ur_joint_dict(joint_dict=ur_joint_state)
        #computes finger pose through forward kinematics and stores it in the state dictionary
        pose, ee_orientation=self.kinematics.forward_kin(ur_joint_state.get_arm_joints_value())
        self.state["gripper_pose"]=np.array(pose)

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
    
    def update_gripper_pose_gazebo(self, new_gripper_pose_gazebo):
        """
        Updates the gripper pose array
        
        Args:
            new_gripper_pose_gazebo (array like): new gripper pose info #x y z r p y

        Returns:
            none
        """
        self.state["gripper_pose_gazebo"]=np.array(new_gripper_pose_gazebo)

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

        return self.state["destination_pose"][0:3] - self.state["gripper_pose_gazebo"][0:3]

    def _get_object_to_target(self):
        """
        Returns the object position in relation to gripper
        """

        return self.state["destination_pose"][0:3] - self.state["cubes_pose"][0, 1:4]

    def to_array(self):
        """
        Retrieves the current state as a list. The order is: ( ur_j_pos + ur_j_vel + gripper_pose_gazebo + gripper_to_obj_dist + cubes_pose + destination_pose)
        The ur_j_pos and ur_j_vel are displayed in standard order (from base to end effector). Cubes pose ignores index 0 = cube id
        
        Args:
            None

        Returns:
            env_array (list): ordered list containing the current environment's state. The array includes the following: target_polar + ur_j_pos (std order) + ur_j_vel (std_order)+ gripper_pose_gazebo + cubes_pose + destination_pose
            for the cubes_pose, the id is ignored [1:]
        """
        gripper_to_obj_pose = self._get_target_to_gripper()

        #env_array= self.state["ur_j_pos"].get_values_std_order().tolist() + self.state["ur_j_vel"].get_values_std_order().tolist() + self.state["gripper_pose_gazebo"].tolist() + gripper_to_obj_pose.tolist() + self.state["cubes_pose"][:, 1:].reshape(-1).tolist() + self.state["destination_pose"].tolist()
        env_array= self.state["ur_j_pos"].get_values_std_order().tolist() + self.state["ur_j_vel"].get_values_std_order().tolist() + self.state["gripper_pose_gazebo"].tolist() + gripper_to_obj_pose.tolist() + self.state["cubes_pose"][:, 1:].reshape(-1).tolist() + self.state["destination_pose"].tolist()
        return env_array

    def get_obs(self):
        gripper_to_obj_pose = self._get_target_to_gripper()
        '''
        obs=dict(
            #where to put the gripper? in the cube to reach
            desired_goal = self.state["destination_pose"][0:3].reshape(-1) ,
            #where the gripper really is
            achieved_goal= self.state["gripper_pose_gazebo"][0:3].reshape(-1) ,
            observation  = np.concatenate([self.state["ur_j_pos_norm"].get_values_std_order()[0:1], self.state["ur_j_vel"].get_values_std_order()[0:1], self.state["gripper_pose_gazebo"][0:3], gripper_to_obj_pose, self.state["cubes_pose"][:, 1:].reshape(-1)]).reshape(-1)
        )
        '''
        cube_to_destination=self._get_object_to_target()
        obs=dict(
            #where to put the gripper? in the cube to reach
            desired_goal = self.state["destination_pose"][0:3].reshape(-1) ,
            #where the gripper really is
            achieved_goal= self.state["cubes_pose"][0, 1:4].reshape(-1) ,
            #observation  = np.concatenate([self.state["gripper_pose"], self.state["cubes_pose"][:, 1:].reshape(-1)]).reshape(-1)
            #gripper pose (1*3); cube to destination(1*3); cubes_pose(1*3); orientation; height
            #observation  = np.concatenate([self.state["gripper_pose"], cube_to_destination, self.state["cubes_pose"][0, 1:4].reshape(-1), self.state["cubes_pose"][0, 6].reshape(-1), self.state["cubes_pose"][0, 9].reshape(-1)]).reshape(-1) #só um cubo
            #gripper pose (1*3); cube to destination(1*3); cubes_pose(1*3)
            observation  = np.concatenate([self.state["gripper_pose"], cube_to_destination, self.state["cubes_pose"][0, 1:4].reshape(-1)]).reshape(-1) #só um cubo
        
        )
        return obs["observation"]

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

class server_state():
    """
    Encapsulates the robot server state
    Includes:
        * ur_j_pos-> robots' joint angles in a ur_joint_dict
        * ur_j_vel-> robots' joint velocities in a ur_joint_dict
        * collision-> array len=1
        * gripper_pose_gazebo-> array len=6 x, y, z, r, p, y 
        * cubes_pose-> array len=7 #id, x, y, z, r, p, y,width,->depth, -height

    """
    
    def __init__(self):
        """
        Populates the structure with:
            * ur_j_pos          (ur_joint_dict) : joint positions in angles (with zeros)
            * ur_j_vel          (ur_joint_dict) : joint velocities (with zeros)
            * collision         (np.array)      : collision array
            * gripper_pose_gazebo  (np.array)   : where is the gripper in the world frame? #id xyzrpy
            * cubes_pose        (np.array)      : where are the cubes? #id xyzrpy
            * destination_pose  (np.array)      : where to move the cubes? xyzrpy
        """
        
        self.state={
            "ur_j_pos": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "ur_j_vel": ur_utils.UR5ROBOTIQ().ur_joint_dict(),
            "collision": np.zeros(1, dtype=np.float32),
            "gripper_pose_gazebo": np.zeros(6, dtype=np.float32), #xyzrpy
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
        '''
        for i in range(len(joints)):
            while joints[i] > np.pi:
                joints[i]-=np.pi
            while joints[i] < -np.pi:
                joints[i]+=np.pi
        '''
        
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
    
    def update_gripper_pose_gazebo(self, new_gripper_pose_gazebo):
        """
        Updates the gripper pose array
        
        Args:
            new_gripper_pose_gazebo (array like): new gripper pose info #x y z r p y

        Returns:
            none
        """
        self.state["gripper_pose_gazebo"]=np.array(new_gripper_pose_gazebo)

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
        e= d + len(self.state["gripper_pose_gazebo"])

        #copies info in the appropriate format
        self.update_ur_joint_pos(      ur_utils.UR5ROBOTIQ().ur_joint_dict().set_values_ros_order(msg[ a:b ]) )
        self.update_ur_joint_vel(      ur_utils.UR5ROBOTIQ().ur_joint_dict().set_values_ros_order(msg[ b:c ] ))
        self.update_collision(         msg[ c:d ] )
        self.update_gripper_pose_gazebo(      msg[ d:e ] )
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
        new_env_state.update_gripper_pose_gazebo(self.state["gripper_pose_gazebo"])
        #consider all cubes
        new_env_state.update_cubes_pose(self.state["cubes_pose"], self.number_of_cubes)
        new_env_state.update_destination_pose(self.state["destination_pose"])
        new_env_state.update_collision(self.state["collision"])
        
        return new_env_state

class GripperReachOpenUR5(UR5RobotiqEnv):
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

class GripperReachOpenUR5Sim(GripperReachOpenUR5, Simulation):
    #cmd = "roslaunch ur_robot_server ur5Robotiq_sim_robot_server.launch \
    #    max_velocity_scale_factor:=0.2 \
    #    action_cycle_rate:=20"


    cmd = "roslaunch ur_robot_server ur5Robotiq_sim_robot_server.launch \
        max_velocity_scale_factor:=0.8 \
        action_cycle_rate:=20 \
        world_name:=one_cube.world \
        rviz_gui:=false \
        gazebo_gui:=true"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        GripperReachOpenUR5.__init__(self, rs_address=self.robot_server_ip, max_episode_steps=1000, robotiq=85, **kwargs)