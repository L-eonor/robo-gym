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



class kinematics_model():
    def __init__(self, ur_model='ur5', gripper_offset=0.15):

        #DH params
        #https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

        #angles (rads)
        self.theta=np.zeros(6, dtype=np.float32) #thetas are the joint variables
        self.alpha=np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0], dtype=np.float32)
        self.number_of_joints=6

        #distance (m)
        if (ur_model=='ur5'):
            self.d=np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823 + gripper_offset], dtype=np.float32)
            self.a=np.array([0 ,-0.425 ,-0.39225 ,0 ,0 ,0], dtype=np.float32)

            #given by the urdf, ofsets configured
            #0], dtype=np.float32)#
            #-np.pi/2
            self.theta_offsets=np.array([np.pi, 0, 0, 0, 0, 0], dtype=np.float32)
            self.alpha_offsets=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
            self.alpha=self.alpha-self.alpha_offsets

            #the robots origin: where is it placed? It is 0.1m above the ground from the urdf files
            self.origin=np.array([0, 0, 0.1], dtype=np.float32)
            #self.wrist_offset=np.array([np.pi/2, np.pi/2, -np.pi])

        elif (ur_model=='ur10'):
            self.d=np.array([0.1273, 0, 0, 0.163941, 0.1157, 0.0922 + gripper_offset], dtype=np.float32)
            self.a=np.array([0 ,-0.612 ,-0.5723 ,0 ,0 ,0], dtype=np.float32)

            #given by the urdf, ofsets configured. THIS IS NOT CORRECTED
            self.theta_offsets=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
            self.alpha_offsets=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
            self.alpha=-self.alpha-self.alpha_offsets

            self.origin=np.array([0, 0, 0], dtype=np.float32)
            #self.wrist_offset=np.array([0, 0, 0])

        else:
            print("error: Invalid ur model")
            raise NotImplementedError

    ################################################################
    #   Forward kinematics
    ################################################################
    def homogeneous_transformation_i(self, theta_i, alpha_i, a_i, d_i):
        """
        Returns the homogeneous transformation A for the joint i, between 2 consecutive links
        Args
            theta_i, alpha_i, a_i, d_i-> variables for the joint i (float)
        Returns
            A_i (np array 4*4)-> homogeneous transformation for joint i
        """

        A_i=np.array(
            [np.cos(theta_i), -np.sin(theta_i)*np.cos(alpha_i),  np.sin(theta_i)*np.sin(alpha_i), a_i*np.cos(theta_i), \
             np.sin(theta_i),  np.cos(theta_i)*np.cos(alpha_i), -np.cos(theta_i)*np.sin(alpha_i), a_i*np.sin(theta_i), \
             0              ,                  np.sin(alpha_i),                  np.cos(alpha_i), d_i                , \
             0              ,                                0,                                0, 1                    ], dtype=np.float32)
        A_i=np.reshape(A_i, (4, 4))

        return A_i

    def homogeneous_transformation_i_var_theta(self, i, theta_i):
        """
        Returns the homogeneous transformation A for the revolute joint i, between 2 consecutive links, where theta is the variable
        Args
            i (int)-> joint order
            theta_i (float)-> value of the joint
        Returns
            A_i (np array 4*4)-> homogeneous transformation for joint i
        """
        return self.homogeneous_transformation_i(theta_i=theta_i, alpha_i=self.alpha[i], a_i=self.a[i], d_i=self.d[i])

    def homogeneous_transformation_i_var_alpha(self, i, alpha_i):
        """
        Returns the homogeneous transformation A for the revolute joint i, between 2 consecutive links, where alpha is the variable
        Args
            i (int)-> joint order
            alpha_i (float)-> value of the joint
        Returns
            A_i (np array 4*4)-> homogeneous transformation for joint i
        """
        return self.homogeneous_transformation_i(theta_i=self.theta[i], alpha_i=self.alpha_i, a_i=self.a[i], d_i=self.d[i])

    def homogeneous_transformation_i_var_a(self, i, a_i):
        """
        Returns the homogeneous transformation A for the prismatic joint i, between 2 consecutive links, where "a" is the variable
        Args
            i (int)-> joint order
            a_i (float)-> value of the joint
        Returns
            A_i (np array 4*4)-> homogeneous transformation for joint i
        """
        return self.homogeneous_transformation_i(theta_i=self.theta[i], alpha_i=self.alpha[i], a_i=self.a_i, d_i=self.d[i])

    def homogeneous_transformation_i_var_d(self, i, d_i):
        """
        Returns the homogeneous transformation A for the prismatic joint i, between 2 consecutive links, where "d" is the variable
        Args
            i (int)-> joint order
            d_i (float)-> value of the joint
        Returns
            A_i (np array 4*4)-> homogeneous transformation for joint i
        """
        return self.homogeneous_transformation_i(theta_i=self.theta[i], alpha_i=self.alpha[i], a_i=self.a[i], d_i=self.d_i)

    def get_transformation_matrix(self, joint_variables, start_joint, end_joint):
        """
        Returns the transformation matrix between base link and end link
        Args
            joint_variables(vector with all joints)-> thetas that are variable
            start_joint (int)-> joint to be considered the base of transformation
            end_joint (int)-> joint of the new base
        Returns
            T_matrix->  transformation matrix (4*4)
        """
        T_matrix=np.eye(4)

        assert end_joint>=start_joint

        joint_index=start_joint
        while joint_index <= end_joint:
            theta_i=joint_variables[joint_index]
            A_i=self.homogeneous_transformation_i_var_theta(joint_index, theta_i)
            T_matrix=np.matmul(T_matrix, A_i)

            joint_index+=1

        return T_matrix

    def forward_kin(self, joint_variables):

        """
        Returns the pose and orientation of the end effector, given the joint values
        Args
            joint_variables(vector with all joints)-> thetas that are variable
        Returns
            pose->  coordinates of the ee frame (x, y, z)
            ee_orientation-> the orientation: x-> normal, y-> sliding (fingers movement), z-> approach(direction to object)

        """

        joints_without_offset=joint_variables-self.theta_offsets
        T_matrix=self.get_transformation_matrix(joint_variables=joints_without_offset, start_joint=0, end_joint=self.number_of_joints-1)
        #print("pose_before")
        #print(T_matrix[0:3, -1])
        pose=np.array(T_matrix[0:3, -1]) + self.origin
        ee_orientation=np.array(T_matrix[0:3, 0:3]) 

        return pose, ee_orientation


    ################################################################
    #   Inverse kinematics
    ################################################################
    
    def inverse_kin(self, pose, orientation):

        """
        Returns the possible joint combinations that results in the pose and orientation defined
        Args
            pose->  coordinates of the ee frame (x, y, z)
            ee_orientation(3*3)-> the orientation: x-> normal, y-> sliding (fingers movement), z-> approach(direction to object)
        Returns
            joints_ik (# of possibilities x 6 joints)-> lines of possible combinations. each line is a combination of joints and each column represents one joint.
            Joints are in std order

        """
        #removes offset to make calculations correct
        pose=pose-self.origin
        #other forms of the pose representation:
        #only the position, but in the vector form (with the final 1)
        pose_vector_form=np.reshape([pose[0], pose[1], pose[2], 1], (-1, 1))
        #matrix that enflobes the orientation and the pose
        pose_full_form=np.hstack((np.vstack((orientation, np.zeros((1, 3)))), pose_vector_form))

        #one line means 1 option for each joint (6 joints), there are 8 possibilities for the same ee pose
        self.joints_ik=np.zeros((8, 6), dtype=np.float32) #thetas are the joint variables

        #####################
        # theta 1 (index 0) #
        #####################

        #computes wrist center-> frame5
        wrist_center=np.array(pose-self.d[5]*orientation[:, 2], dtype=np.float32)
        wrist_x=wrist_center[0]
        wrist_y=wrist_center[1]
        wrist_r=np.sqrt(wrist_x**2 + wrist_y**2)
        
        psi = np.arctan2(wrist_y, wrist_x)
        phi = np.arccos(self.d[3] /wrist_r)
        #solutions for theta1, either left or right shoulder
        theta1_possible_values=np.array([np.pi/2 + psi + phi, np.pi/2 + psi - phi], dtype=np.float32)
        self.joints_ik[0:4, 0]=theta1_possible_values[0] 
        self.joints_ik[4: , 0]=theta1_possible_values[1] 

        #####################
        # theta 5 and 6 (index 4 and 5) #
        #####################

        theta5_possible_values=np.zeros(2)
        theta6_possible_values=np.zeros(2*2)

        for i in range(len(theta1_possible_values)):
            #homogeneouos transformation between frames 0 and 1
            T_01=self.get_transformation_matrix(joint_variables=[theta1_possible_values[i], 0, 0, 0, 0, 0], start_joint=0, end_joint=0)
            #homogeneous transformation 1->0 to obtain desired point in the frame 1
            T_10=np.linalg.inv(T_01)
            #desired pose and orientation in the frame 1
            pose_frame1=np.matmul(T_10, pose_vector_form)
            pose_full_form_frame1=np.matmul(T_10, pose_full_form)

            start_indice=i*4

            #theta5
            theta5_possible_values[i]=np.arccos((pose_frame1[2]-self.d[3])/self.d[5])
            self.joints_ik[start_indice  :start_indice+2, 4]=  theta5_possible_values[i]
            self.joints_ik[start_indice+2:start_indice+4, 4]= -theta5_possible_values[i]

            #theta6
            theta6_possible_values[i*2  ]=np.arctan2(  -pose_full_form_frame1[2, 1] / np.sin( theta5_possible_values[i]), pose_full_form_frame1[2, 0] / np.sin( theta5_possible_values[i])      )
            theta6_possible_values[i*2+1]=np.arctan2(  -pose_full_form_frame1[2, 1] / np.sin(-theta5_possible_values[i]), pose_full_form_frame1[2, 0] / np.sin(-theta5_possible_values[i])      )
            self.joints_ik[start_indice  :start_indice+2, 5]= theta6_possible_values[i*2  ]
            self.joints_ik[start_indice+2:start_indice+4, 5]= theta6_possible_values[i*2+1]

     

        #####################
        # theta 3 (index 2) #
        #####################
        impossible_combinations=[]
        for combination_theta1_theta6 in range(len(theta6_possible_values)):
            hypothesis_index=2*combination_theta1_theta6

            ### To compute desired pose in frame 4
            #homogeneouos transformation between frames 0 and 1
            T_01=self.get_transformation_matrix(joint_variables=self.joints_ik[hypothesis_index, :], start_joint=0, end_joint=0)
            #homogeneous transformation 1->0 to obtain desired point in the frame 1
            T_10=np.linalg.inv(T_01)
            #desired pose and orientation in the frame 1
            pose_full_form_frame1=np.matmul(T_10, pose_full_form)
            #transformation that converts from base 4 to frame 6 (T64)
            #homogeneouos transformation between frames 5 and 6
            T_56=self.get_transformation_matrix(joint_variables=self.joints_ik[hypothesis_index, :], start_joint=5, end_joint=5)
            #homogeneouos transformation between frames 4 and 5
            T_45=self.get_transformation_matrix(joint_variables=self.joints_ik[hypothesis_index, :], start_joint=4, end_joint=4)
            T_64=np.linalg.inv(np.matmul(T_45,T_56))

            # desired pose in frame 4
            T_14=np.matmul(pose_full_form_frame1, T_64)
            # desired pose in frame 3
            P_13=np.matmul(T_14,np.resize([0, -self.d[3], 0, 1], (4, 1)))[0:3]

            #theta 3
            coeff=(np.linalg.norm(P_13)**2 - self.a[1]**2 - self.a[2]**2 )/(2 * self.a[1] * self.a[2])
            if coeff>1 or coeff<-1:
                theta3=np.nan
                #signals the lines to be removed 
                impossible_combinations.insert(0, hypothesis_index)
                impossible_combinations.insert(0, hypothesis_index+1)
            else:
                theta3 = np.arccos ((np.linalg.norm(P_13)**2 - self.a[1]**2 - self.a[2]**2 )/(2 * self.a[1] * self.a[2]))
            self.joints_ik[hypothesis_index  , 2] =  theta3
            self.joints_ik[hypothesis_index+1, 2] = -theta3

        #deletes impossible hypotesis
        possible_joints=copy.deepcopy(self.joints_ik)
        for i in impossible_combinations:
            possible_joints=np.delete(possible_joints, i, axis=0)
        self.joints_ik=copy.deepcopy(possible_joints)
            
        #####################
        # theta 2 and 4 (index 1 and 3) #
        #####################

        for hypothesis_index in range(self.joints_ik.shape[0]):

            ### To compute desired pose in frame 4
            #homogeneouos transformation between frames 0 and 1
            T_01=self.get_transformation_matrix(joint_variables=self.joints_ik[hypothesis_index, :], start_joint=0, end_joint=0)
            #homogeneous transformation 1->0 to obtain desired point in the frame 1
            T_10=np.linalg.inv(T_01)
            #desired pose and orientation in the frame 1
            pose_full_form_frame1=np.matmul(T_10, pose_full_form)
            #homogeneouos transformation between frames 5 and 6
            T_56=self.get_transformation_matrix(joint_variables=self.joints_ik[hypothesis_index, :], start_joint=5, end_joint=5)
            #homogeneous transformation 6->5 to obtain desired point in the frame 6
            T_65=np.linalg.inv(T_56)
            #homogeneouos transformation between frames 4 and 5
            T_45=self.get_transformation_matrix(joint_variables=self.joints_ik[hypothesis_index, :], start_joint=4, end_joint=4)
            #homogeneous transformation 5->4 to obtain desired point in the frame 1
            T_54=np.linalg.inv(T_45)

            # desired pose in frame 4
            T_14=np.matmul(np.matmul(pose_full_form_frame1, T_65), T_54)
            # desired pose in frame 3
            P_13=np.matmul(T_14,np.resize([0, -self.d[3], 0, 1], (4, 1)))[0:3]


            
            # theta 2
            self.joints_ik[hypothesis_index, 1]= -np.arctan2(P_13[1], -P_13[0]) + np.arcsin(self.a[2]*np.sin(self.joints_ik[hypothesis_index, 2])/np.linalg.norm(P_13))

            # theta 4
            #homogeneouos transformation between frames 2 and 3
            T_23=self.get_transformation_matrix(joint_variables=self.joints_ik[hypothesis_index, :], start_joint=2, end_joint=2)
            #homogeneous transformation 3->2 to obtain desired point in the frame 3
            T_32=np.linalg.inv(T_23)
            #homogeneouos transformation between frames 1 and 2
            T_12=self.get_transformation_matrix(joint_variables=self.joints_ik[hypothesis_index, :], start_joint=1, end_joint=1)
            #homogeneous transformation 2->1 to obtain desired point in the frame 2
            T_21=np.linalg.inv(T_12)

            T_34 = np.matmul(np.matmul(T_32 , T_21), T_14)
            self.joints_ik[hypothesis_index, 3]=np.arctan2(T_34[1,0], T_34[0,0])


        #removes lines containing nan values
        self.joints_ik=self.joints_ik[~np.isnan(self.joints_ik).any(axis=1)]

        #compensates offsets
        self.joints_ik=self.joints_ik - np.repeat([self.theta_offsets], self.joints_ik.shape[0], axis=0)
        return self.joints_ik
       
    ################################################################
    #   choose inverse kin possibility
    ################################################################ 

    def get_joint_combination (self, pose, orientation, current_joints):
        """
        Returns one joint combination to achieve the input pose and orientation. If returns -1 there are no valid/possible combinations

        Args
            pose->  coordinates of the ee frame (x, y, z)
            ee_orientation(3*3)-> the orientation: x-> normal, y-> sliding (fingers movement), z-> approach(direction to object)
            current_joints(1*6)-> current robot joints, in std order
        Returns
            chosen_combination(1*6)-> possible and valid combination of joints. Returns None if it is impossible
        """
        #orientation=np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        #orientation=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        possible_joints=self.inverse_kin(pose=pose, orientation=orientation)
        #print("possible_joints")
        #print(possible_joints)

        #there are any possible combinations?
        if len(possible_joints)==0:
            return None

        #Evaluates every valid combination: computes joint centers and finds if they are above the ground (z>0)
        valid_joints=[]
        for joint_combination in possible_joints:
            joints_without_offset=joint_combination-self.theta_offsets

            #computes joint centers and tests z coordinate
            T_matrix=np.eye(4)
            for joint_index in range(len(joints_without_offset)):
                theta_i=joints_without_offset[joint_index]
                A_i=self.homogeneous_transformation_i_var_theta(joint_index, theta_i)
                T_matrix=np.matmul(T_matrix, A_i)

                frame_center=T_matrix[0:3, -1]
                frame_center_with_correction=frame_center + self.origin
                frame_center_z_coord=frame_center_with_correction[-1]

                #if joint is below ground, delete possibility
                if(frame_center_z_coord<0):
                    break
                #all all joints from the combination are valid, append to valid joints
                if(joint_index==len(joints_without_offset)-1):
                    if(len(valid_joints)==0):
                        valid_joints=np.reshape(joint_combination, (1, len(joint_combination)))
                    else:
                        valid_joints=np.vstack((valid_joints, joint_combination))
                        
        #there are any valid combinations?
        if len(valid_joints)==0:
            return None

        #Evaluates the valid joints in relation to the current joints; computes the difference and picks up the most similar combination
        #difference between each possibility and current joints
        joint_difference=np.abs(valid_joints-current_joints)
        total_difference_per_combination=np.sum(joint_difference, axis=1)
        #picks up the most similar combination
        chosen_combination=valid_joints[np.argmin(total_difference_per_combination), :]






    
    def normaliza_pi(self, x):
        normalized=copy.deepcopy(x)
        x=x.flatten()
        for i in range(len(x)):
            if x[i]>np.pi:
                while x[i]>np.pi:
                    x[i]-=np.pi
            elif x[i]<-np.pi:
                while x[i]<-np.pi:
                    x[i]+=np.pi
        return np.reshape(x, normalized.shape )
