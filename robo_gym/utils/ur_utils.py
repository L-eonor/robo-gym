#!/usr/bin/env python3

import numpy as np
import copy

class UR5():
    """Universal Robots UR5 utilities.

    Attributes:
        max_joint_positions (np.array): Description of parameter `max_joint_positions`.
        min_joint_positions (np.array): Description of parameter `min_joint_positions`.
        max_joint_velocities (np.array): Description of parameter `max_joint_velocities`.
        min_joint_velocities (np.array): Description of parameter `min_joint_velocities`.

    """
    def __init__(self):

        # Indexes go from shoulder pan joint to end effector
        self.max_joint_positions = np.array([6.28,6.28,6.28,6.28,6.28,6.28])
        self.min_joint_positions = - self.max_joint_positions
        self.max_joint_velocities = np.array([np.inf] * 6)
        self.min_joint_velocities = - self.max_joint_velocities

    def _ros_joint_list_to_ur5_joint_list(self,ros_thetas):
        """Transform joint angles list from ROS indexing to standard indexing.

        Rearrange a list containing the joints values from the joint indexes used
        in the ROS join_states messages to the standard joint indexing going from
        base to end effector.

        Args:
            ros_thetas (list): Joint angles with ROS indexing.

        Returns:
            np.array: Joint angles with standard indexing.

        """

        return np.array([ros_thetas[2],ros_thetas[1],ros_thetas[0],ros_thetas[3],ros_thetas[4],ros_thetas[5]])

    def _ur_5_joint_list_to_ros_joint_list(self,thetas):
        """Transform joint angles list from standard indexing to ROS indexing.

        Rearrange a list containing the joints values from the standard joint indexing
        going from base to end effector to the indexing used in the ROS
        join_states messages.

        Args:
            thetas (list): Joint angles with standard indexing.

        Returns:
            np.array: Joint angles with ROS indexing.

        """

        return np.array([thetas[2],thetas[1],thetas[0],thetas[3],thetas[4],thetas[5]])

    def get_random_workspace_pose(self):
        """Get pose of a random point in the UR5 workspace.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        pose =  np.zeros(6)

        singularity_area = True

        # check if generated x,y,z are in singularityarea
        while singularity_area:
            # Generate random uniform sample in semisphere taking advantage of the
            # sampling rule

            # UR5 workspace radius
            # Max d = 1.892
            R =  0.900 # reduced slightly

            phi = np.random.default_rng().uniform(low= 0.0, high= 2*np.pi)
            costheta = np.random.default_rng().uniform(low= 0.0, high= 1.0) # [-1.0,1.0] for a sphere
            u = np.random.default_rng().uniform(low= 0.0, high= 1.0)

            theta = np.arccos(costheta)
            r = R * np.cbrt(u)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            if (x**2 + y**2) > 0.085**2:
                singularity_area = False

        pose[0:3] = [x,y,z]

        return pose

    def get_max_joint_positions(self):

        return self.max_joint_positions

    def get_min_joint_positions(self):

        return self.min_joint_positions

    def get_max_joint_velocities(self):

        return self.max_joint_velocities

    def get_min_joint_velocities(self):

        return self.min_joint_velocities

    def normalize_joint_values(self, joints):
        """Normalize joint position values
        
        Args:
            joints (np.array): Joint position values

        Returns:
            norm_joints (np.array): Joint position values normalized between [-1 , 1]
        """
        for i in range(len(joints)):
            if joints[i] <= 0:
                joints[i] = joints[i]/abs(self.min_joint_positions[i])
            else:
                joints[i] = joints[i]/abs(self.max_joint_positions[i])
        return joints

class UR10():
    """Universal Robots UR10 utilities.

    Attributes:
        max_joint_positions (np.array): Description of parameter `max_joint_positions`.
        min_joint_positions (np.array): Description of parameter `min_joint_positions`.
        max_joint_velocities (np.array): Description of parameter `max_joint_velocities`.
        min_joint_velocities (np.array): Description of parameter `min_joint_velocities`.

    """
    def __init__(self):

        # Indexes go from shoulder pan joint to end effector
        self.max_joint_positions = np.array([6.28,6.28,3.14,6.28,6.28,6.28])
        self.min_joint_positions = - self.max_joint_positions
        self.max_joint_velocities = np.array([np.inf] * 6)
        self.min_joint_velocities = - self.max_joint_velocities

    def _ros_joint_list_to_ur10_joint_list(self,ros_thetas):
        """Transform joint angles list from ROS indexing to standard indexing.

        Rearrange a list containing the joints values from the joint indexes used
        in the ROS join_states messages to the standard joint indexing going from
        base to end effector.

        Args:
            ros_thetas (list): Joint angles with ROS indexing.

        Returns:
            np.array: Joint angles with standard indexing.

        """

        return np.array([ros_thetas[2],ros_thetas[1],ros_thetas[0],ros_thetas[3],ros_thetas[4],ros_thetas[5]])

    def _ur_10_joint_list_to_ros_joint_list(self,thetas):
        """Transform joint angles list from standard indexing to ROS indexing.

        Rearrange a list containing the joints values from the standard joint indexing
        going from base to end effector to the indexing used in the ROS
        join_states messages.

        Args:
            thetas (list): Joint angles with standard indexing.

        Returns:
            np.array: Joint angles with ROS indexing.

        """

        return np.array([thetas[2],thetas[1],thetas[0],thetas[3],thetas[4],thetas[5]])


    def normalize_joint_values(self, joints):
        """Normalize joint position values
        
        Args:
            joints (np.array): Joint position values

        Returns:
            norm_joints (np.array): Joint position values normalized between [-1 , 1]
        """
        for i in range(len(joints)):
            if joints[i] <= 0:
                joints[i] = joints[i]/abs(self.min_joint_positions[i])
            else:
                joints[i] = joints[i]/abs(self.max_joint_positions[i])
        return joints


    
    def get_random_workspace_pose(self):
        """Get pose of a random point in the UR10 workspace.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        pose =  np.zeros(6)

        singularity_area = True

        # check if generated x,y,z are in singularityarea
        while singularity_area:
            # Generate random uniform sample in semisphere taking advantage of the
            # sampling rule

            # UR10 workspace radius
            # Max d = 2.547
            R =  1.200 # reduced slightly

            phi = np.random.default_rng().uniform(low= 0.0, high= 2*np.pi)
            costheta = np.random.default_rng().uniform(low= 0.0, high= 1.0) # [-1.0,1.0] for a sphere
            u = np.random.default_rng().uniform(low= 0.0, high= 1.0)

            theta = np.arccos(costheta)
            r = R * np.cbrt(u)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            if (x**2 + y**2) > 0.095**2:
                singularity_area = False

        pose[0:3] = [x,y,z]

        return pose

    def get_max_joint_positions(self):

        return self.max_joint_positions

    def get_min_joint_positions(self):

        return self.min_joint_positions

    def get_max_joint_velocities(self):

        return self.max_joint_velocities

    def get_min_joint_velocities(self):

        return self.min_joint_velocities

class UR5ROBOTIQ():
    """Universal Robots UR5 + Robotiq utilities.-> requires a new class because there's a new joint in /joint_states

    Attributes:
        max_joint_positions (np.array): Description of parameter `max_joint_positions`.
        min_joint_positions (np.array): Description of parameter `min_joint_positions`.
        max_joint_velocities (np.array): Description of parameter `max_joint_velocities`.
        min_joint_velocities (np.array): Description of parameter `min_joint_velocities`.

    """

    
    def __init__(self, robotiq=85):

        #joint names
        self.joint_names_rostopic=self.ur_joint_dict().get_names_ros_order() #joint names according to /joint_states order                                    
        self.joint_names_standard=self.ur_joint_dict().get_names_std_order() #joint names according to /joint_states order
        
        #joint number of ..
        self.number_of_joints = len(self.joint_names_rostopic)
        self.number_of_joint_positions = self.number_of_joints
        self.number_of_joint_velocities = self.number_of_joints

        
        # joint positions, standard order
        # Indexes go from shoulder pan joint to end effector
        if (robotiq==85): #if robotiq 85, finger joint lims are 0 and 0.8
            #max_joint_pos=   np.array([6.28,6.28,6.28,6.28,6.28,6.28, 0.8])
            #min_joint_pos= - np.array([6.28,6.28,6.28,6.28,6.28,6.28, 0])
            max_joint_pos=  np.array([ np.pi, -2*np.pi/5,  2.7,      0,  np.pi,  np.pi, 0.8])
            min_joint_pos=  np.array([-np.pi, -3*np.pi/5, -2.7, -np.pi, -np.pi, -np.pi, 0])
        elif (robotiq==140): #if robotiq 140, finger joint lims are 0 and 0.7
            #max_joint_pos =   np.array([6.28,6.28,6.28,6.28,6.28,6.28, 0.7])
            #min_joint_pos = - np.array([6.28,6.28,6.28,6.28,6.28,6.28, 0])
            max_joint_pos=  np.array([ np.pi, -2*np.pi/5,  2.7,      0,  np.pi,  np.pi, 0.7])
            min_joint_pos=  np.array([-np.pi, -3*np.pi/5, -2.7, -np.pi, -np.pi, -np.pi, 0])

        else:
            raise InvalidStateError('Invalid gripper')
        # updates objects
        self.ur_joint_max_pos = self.ur_joint_dict().set_values_std_order(max_joint_pos)
        self.ur_joint_min_pos = self.ur_joint_dict().set_values_std_order(min_joint_pos)

        #joint velocities, standard order
        self.ur_joint_max_vel = self.ur_joint_dict().set_values_std_order( np.array([np.inf] * self.number_of_joint_velocities))
        self.ur_joint_min_vel = self.ur_joint_dict().set_values_std_order(-np.array([np.inf] * self.number_of_joint_velocities))

    def get_random_workspace_pose(self):
        """Get pose of a random point in the UR5 workspace.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        pose =  np.zeros(6)

        singularity_area = True

        # check if generated x,y,z are in singularityarea
        while singularity_area:
            # Generate random uniform sample in semisphere taking advantage of the
            # sampling rule

            # UR5 workspace radius
            # Max d = 1.892
            R =  0.900 # reduced slightly

            phi = np.random.default_rng().uniform(low= 0.0, high= 2*np.pi)
            costheta = np.random.default_rng().uniform(low= 0.0, high= 1.0) # [-1.0,1.0] for a sphere
            u = np.random.default_rng().uniform(low= 0.0, high= 1.0)

            theta = np.arccos(costheta)
            r = R * np.cbrt(u)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            if (x**2 + y**2) > 0.085**2:
                singularity_area = False

        pose[0:3] = [x,y,z]

        return pose

    def get_max_joint_positions(self):

        return self.ur_joint_max_pos

    def get_min_joint_positions(self):

        return self.ur_joint_min_pos

    def get_max_joint_velocities(self):

        return self.ur_joint_max_vel

    def get_min_joint_velocities(self):

        return self.ur_joint_min_vel

    def normalize_ur_joint_dict(self, joint_dict):
        """Normalize joint position values
        
        Args:
            joints (ur_joint_dict): Joint object (std order) 

        Returns:
            norm_joints (ur_joint_dict): Joint position object with values normalized between [-1 , 1]
        """
        joints = joint_dict.get_values_std_order()
        normalized_joints = copy.deepcopy(joints)
        #max value a joint can take
        max_joints = self.ur_joint_max_pos.get_values_std_order()
        #min value a joint can take
        min_joints = self.ur_joint_min_pos.get_values_std_order()
        #joint range and auxiliar vars to calculate normalized values
        joint_range = max_joints - min_joints
        joint_delta = joint_range/2
        central_joint_value=min_joints + joint_range/2
        
        #joint normalization, results in values between -1 and 1
        normalized_joints= (joints - central_joint_value)/joint_delta
        #convert to dict obj
        new_joints_dict=self.ur_joint_dict().set_values_std_order(normalized_joints)

        return new_joints_dict




    class ur_joint_dict:
        """ 
        Saves the joint values (position or velocity) associated with each joint name. Values are saved in a dic, associated to each joint name.
        Tries to overcome the problems between standard order (natural from base joint to end effector) and ros order (alphabetical)

            Standard order, from base to end effector:shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint, finger_joint
            Ros topic message order: elbow_joint, finger_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint

        """

        def __init__(self):

            #joint names in each order
            self.std_order=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "finger_joint"]
            self.ros_order=["elbow_joint", "finger_joint", "shoulder_lift_joint", "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

            #arm joints vs finger joints in standard order
            self.arm_joints=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"] #std order
            self.finger_joints=["finger_joint"] #std order

            #dictionary that associates a joint value (position or velocity) to the respective name
            self.joints={"shoulder_pan_joint"  : 0,
                    "shoulder_lift_joint" : 0,
                    "elbow_joint"         : 0,
                    "wrist_1_joint"       : 0,
                    "wrist_2_joint"       : 0,
                    "wrist_3_joint"       : 0,
                    "finger_joint"        : 0
                    }


        def set_values_ros_order(self, values):
            """
            Sets the joint values according to the array passed in ros order
            Ros topic message order: elbow_joint, finger_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint

                Arguments:
                    * values (array like)- array of ordered joint values (ros order)
                Return:
                    * self (ur_joint_dict)
            """
            values=np.array(values)

            #validates array dimensions
            if len(values) != len(self.joints) :
                raise RobotServerError("Invalid array dimensions")
            
            #assigns correct joints
            for index in range(len(self.joints)):
                self.joints[self.ros_order[index]]=values[index]


            return self

        def set_values_std_order(self, values):
            """
            Sets the joint values according to the array passed in std order
            Standard order (base to end effector): shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint, finger_joint

                Arguments:
                    * values (array like)- array of ordered joint values (std order)
                Return:
                    * self (ur_joint_dict)
            """
            values=np.array(values)

            #validates array dimensions
            if len(values) != len(self.joints) :
                raise RobotServerError("Invalid array dimensions")
            
            #assigns correct joints
            for index in range(len(self.joints)):
                self.joints[self.std_order[index]]=values[index]


            return self

        def get_values_ros_order(self):
            """
            Returns array of joint values in ros order
            Ros topic message order: elbow_joint, finger_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint

                Arguments:
                    * None
                Return:
                    *  (np array): joint values in ros order
            """

            return np.fromiter( [self.joints.get(key) for key in self.ros_order] , dtype=np.float32)
        
        def get_values_std_order(self):
            """
            Returns array of joint values in std order
            Standard order (base to end effector): shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint, finger_joint

                Arguments:
                    * None
                Return:
                    *  (np array): joint values in std order
            """

            return np.fromiter( [self.joints.get(key) for key in self.std_order] , dtype=np.float32)

        def get_names_ros_order(self):
            """
            Returns array of joint names in ros order

                Arguments:
                    * None
                Return:
                    *  (np array): joint names in std order
            """
            
            return self.ros_order

        def get_names_std_order(self):
            """
            Returns array of joint names in std order

                Arguments:
                    * None
                Return:
                    *  (np array): joint names in std order
            """
            return self.std_order

        def get_finger_joints_value(self):
            """
            Returns array of finger joint values in std order

                Arguments:
                    * None
                Return:
                    *  (np array): finger joint values in std order
            """
            return np.fromiter( [self.joints.get(key) for key in self.finger_joints] , dtype=np.float32)

        def get_arm_joints_value(self):
            """
            Returns array of arm joint values in std order

                Arguments:
                    * None
                Return:
                    *  (np array): arm joint values in std order
            """
            return np.fromiter( [self.joints.get(key) for key in self.arm_joints] , dtype=np.float32)

        def get_number_of_joints(self):
            """
            Returns the total number of joints

                Arguments:
                    * None
                Return:
                    * (int): number of joints
            """
            return len(self.joints)

        def get_number_of_arm_joints(self):
            """
            Returns the number of arm joints

                Arguments:
                    * None
                Return:
                    * (int): number of arm joints
            """
            return len(self.get_arm_joints_value())

        def get_number_of_finger_joints(self):
            """
            Returns the number of finger joints

                Arguments:
                    * None
                Return:
                    * (int): number of finger joints
            """
            return len(self.get_finger_joints_value())