#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import WrenchStamped

from std_msgs.msg import Float64, String, Bool

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from rclpy.qos import qos_profile_sensor_data, QoSProfile

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from cv_bridge import CvBridge
import time
import math

# Frame Imports
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper
from sensor_msgs.msg import JointState

# States
INITIALIZATION = 0
FINDING_GROUND = 1
FINDING_DOLLAR = 2
FINDING_SQUARE = 3
PREPPING_GRAB = 4
DUMMY_STATE = 5

# Events
ES_TIMEOUT = 11
ES_COMMAND_EXECUTED = 12
# Pseudo-Events
ES_ITEM_DETECTED = 13
ES_ITEM_UNDETECTED = 14
ES_ITEM_PARTIAL = 15
ES_COLLISION_DETECTED = 16
ES_NO_COLLISION_DETECTED = 17

class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')

        # Use this to set the name of the camera topics depending on whether we're in sim or real
        self.usingRealRobot = True

        # Camera Topic Names
        self.CameraIntrinsicsTopic = "/color/camera_info"
        self.DepthCameraTopic = "/aligned_depth_to_color/image_raw"
        if self.usingRealRobot:
            self.CameraIntrinsicsTopic = "/camera/realsense2_camera_node" + self.CameraIntrinsicsTopic
            self.DepthCameraTopic = "/camera/realsense2_camera_node" + self.DepthCameraTopic
        
        # Publishers
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)
        self.scan_dollar_pub = self.create_publisher(Bool, '/scan_dollar_request', 10)
        self.scan_square_pub = self.create_publisher(Bool, '/scan_square_request', 10)
        self.cancel_command_pub = self.create_publisher(Bool, '/me314_xarm_cancel_command', 10)

        # Subscribers
        self.CameraIntrinicsSubscriber = self.create_subscription(CameraInfo,self.CameraIntrinsicsTopic,self.GetCameraIntrinsics,1) # RGB Camera Intrinsics
        self.depth_camera_subscription = self.create_subscription(Image,self.DepthCameraTopic,self.GetDepthCV2Image,qos_profile=qos_profile_sensor_data) # Depth Camera
        self.depth_camera_subscription  # prevent unused variable warning
        self.CommandSubscriber = self.create_subscription(String,"/me314_xarm_current_command",self.GetCurrentCommand,10) # Current Command in Execution
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.GetCurrentPose, 10) # Gets Current EE Pose
        self.JointAngleSubscriber = self.create_subscription(JointState, '/me314_xarm_current_joint_positions_deg', self.GetJointAngles, 10) # Gets current joint angles
        self.dollar_sub = self.create_subscription(Pose, "/dollar_report", self.DollarCallback, 10) # Gets report regarding dollar
        self.square_sub = self.create_subscription(Point, "/square_report", self.SquareCallback, 10) # Gets report regarding square
        self.ft_ext_state_sub = self.create_subscription(WrenchStamped, '/xarm/uf_ftsensor_ext_states', self.ForceCallback, 10) # Gets current wrench measurement

        # CV Bridge Initialization
        self.bridge = CvBridge()

        # Frame Listener Initialization
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Home Position Joint Angles
        self.home_joints_deg = [0.2, -67.2, -0.2, 24.2, 0.4, 91.4, 0.3]
        self.home_joints_rad = [math.radians(angle) for angle in self.home_joints_deg]

        # Location of Camera Relative to EE
        self.camera_wrt_EE = 0.05

        # Initialization of Various Variables
        self.cv_DepthImage = None
        self.alpha = None
        self.beta = None
        self.u0 = None
        self.v0 = None
        self.haveIntrinsics_RBG = False
        self.haveDepthImage = False
        self.previousCommand = ""
        self.finalCommand = None
        self.baseTransform = None
        self.EE_pos = None
        self.DollarPoint = None
        self.PlacePoint = None
        self.aboveDollar = False
        self.haveWrench = False
        self.currentForce = None
        self.currentTorque = None
        self.ground = None
        self.commandCancelled = False
        self.FirstTimeout = False
        self.state = INITIALIZATION
        self.goHome = True

        # Set a timer to repeatedly check if we have Camera Intrinsics and Depth Camera visuals
        self.timer_SM = self.create_timer(2.0, self.TimeoutCallback)

    
    def GetTransform(self, target_frame, source_frame):
        try:
            #now = self.get_clock().now()
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(seconds=0)
            )
            translate2Base = transform.transform.translation
            rotate2Base_quat = transform.transform.rotation
            #self.get_logger().info(f'Transform from camera_locobot_link to locobot_arm_base_link:')
            #self.get_logger().info(f'Translation: x={translation.x:.2f}, y={translation.y:.2f}, z={translation.z:.2f}')
            #self.get_logger().info(f'Rotation: x={rotation.x:.2f}, y={rotation.y:.2f}, z={rotation.z:.2f}, w={rotation.w:.2f}')
            baseRotationObject = R.from_quat([rotate2Base_quat.x,rotate2Base_quat.y,rotate2Base_quat.z,rotate2Base_quat.w])
            baseRotMat = baseRotationObject.as_matrix()
            transformMatrix = np.zeros((3,4))
            transformMatrix[:,:3] = baseRotMat
            transformMatrix[0,3] = translate2Base.x
            transformMatrix[1,3] = translate2Base.y
            transformMatrix[2,3] = translate2Base.z
            return transformMatrix, True
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {ex}')
            return None, False
    
    def GetCameraIntrinsics(self, CameraMsg):
        self.get_logger().info(f'Getting camera intrinsics')
        self.alpha = CameraMsg.k[0]
        self.beta = CameraMsg.k[4]
        self.u0 = CameraMsg.k[2]
        self.v0 = CameraMsg.k[5]
        self.destroy_subscription(self.CameraIntrinicsSubscriber)
        self.haveIntrinsics_RBG = True
    
    def GetDepthCV2Image(self,DepthImageMsg):
        self.cv_DepthImage = self.bridge.imgmsg_to_cv2(DepthImageMsg, desired_encoding='passthrough')
        self.haveDepthImage = True
    
    def getPointInBaseFrame(self,pixel_x,pixel_y):
        Z_c = self.cv_DepthImage[int(pixel_y),int(pixel_x)]/1000
        X_c = (pixel_x - self.u0)*Z_c/self.alpha
        Y_c = (pixel_y - self.v0)*Z_c/self.beta
        self.get_logger().info(f'Desired object Camera Location: x={X_c:.2f}, y={Y_c:.2f}, z={Z_c:.2f}')
        point_c_H = np.array([X_c,Y_c,Z_c,1])
        point_base = self.baseTransform @ point_c_H
        if point_base[0] > 0.5:
            point_base[0] = 0.5
        elif point_base[0] < 0.15:
            point_base[0] = 0.15
        if point_base[1] > 0.3:
            point_base[1] = 0.3
        elif point_base[1] < -0.3:
            point_base[1] = -0.3
        if point_base[2] > 0.4:
            point_base[2] = 0.4
        elif point_base[2] < 0.01:
            point_base[2] = 0.01
        self.get_logger().info(f'Desired object Base location: x={point_base[0]:.2f}, y={point_base[1]:.2f}, z={point_base[2]:.2f}')
        return point_base
    
    def GetCurrentPose(self, msg: Pose):
        self.EE_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.EE_ori = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
    
    def GetJointAngles(self,JointMessage):
        self.joint_positions = [math.radians(pos) for pos in JointMessage.position]
    
    def GetCurrentCommand(self,command):
        #self.get_logger().info(f'Command = {command.data}')
        if command.data != self.previousCommand:
            if self.finalCommand in self.previousCommand and command.data == "":
                self.StateMachine(ES_COMMAND_EXECUTED)
        self.previousCommand = command.data
    
    def DollarCallback(self,msg):
        if msg.position.z == 1.0:
            self.pixel_x_dollar = msg.position.x
            self.pixel_y_dollar = msg.position.y
            self.dollarAngle = math.radians(msg.orientation.z) + np.pi/2
            if self.dollarAngle > np.pi/2:
                self.dollarAngle -= np.pi
            elif self.dollarAngle < -np.pi/2:
                self.dollarAngle += np.pi
            self.StateMachine(ES_ITEM_DETECTED)
        elif msg.position.z == 0.5:
            self.pixel_x_dollar = msg.position.x
            self.pixel_y_dollar = msg.position.y
            self.StateMachine(ES_ITEM_PARTIAL)
        elif msg.position.z == 0.0:
            self.StateMachine(ES_ITEM_UNDETECTED)
    
    def SquareCallback(self,msg):
        if msg.z == 1.0:
            self.pixel_x_square = msg.x
            self.pixel_y_square = msg.y
            self.StateMachine(ES_ITEM_DETECTED)
        elif msg.z == 0.5:
            self.pixel_x_square = msg.x
            self.pixel_y_square = msg.y
            self.StateMachine(ES_ITEM_PARTIAL)
        elif msg.z == 0.0:
            self.StateMachine(ES_ITEM_UNDETECTED)
    
    def ForceCallback(self,msg):
        self.currentForce = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        self.currentTorque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        self.haveWrench = True
    
    def checkCollision(self):
        # Get the current transformation from the FT Sensor frame to the base frame
        target_frame = 'link_base'
        source_frame = 'ft_sensor_link'
        FT_To_BaseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
        # Don't move onto the next part in the code until we've found the transformation matrix
        while not haveTransform:
            FT_To_BaseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
        baseForce = FT_To_BaseTransform[:,:-1] @ self.currentForce
        if baseForce[2] > 2:
            return ES_COLLISION_DETECTED
        else:
            return ES_NO_COLLISION_DETECTED
    
    def TimeoutCallback(self):
        if not self.FirstTimeout:
            self.FirstTimeout = True
        else:
            self.StateMachine(ES_TIMEOUT)
    
    def CancelTimerSM(self):
        self.timer_SM.cancel()
        self.FirstTimeout = False
    
    def StateMachine(self,Event):
        if self.state == INITIALIZATION:
            if Event == ES_TIMEOUT:
                # The only point of this state is to ensure we have all the camera stuff before we start looking for the dollar
                # This state is entered by a timer timing out every 2 seconds
                self.get_logger().info(f'Checking if camera and force stuff has been obtained')
                if self.haveDepthImage and self.haveIntrinsics_RBG and self.haveWrench:
                    self.get_logger().info(f'Camera and force stuff has been obtained . . . First putting gripper in home position')
                    # Cancel the Timer that continually checks if we have the stuff
                    self.CancelTimerSM()

                    # Create a CommandQueue message
                    queue_msg = CommandQueue()
                    queue_msg.header.stamp = self.get_clock().now().to_msg()

                    # Create a CommandWrapper for the gripper command to open
                    wrapper_gripper_open = CommandWrapper()
                    wrapper_gripper_open.command_type = "gripper"
                    wrapper_gripper_open.gripper_command.gripper_position = 0.0

                    wrapper_home = CommandWrapper()
                    wrapper_home.command_type = "joint"
                    wrapper_home.joint_command.joint1 = self.home_joints_rad[0]
                    wrapper_home.joint_command.joint2 = self.home_joints_rad[1]
                    wrapper_home.joint_command.joint3 = self.home_joints_rad[2]
                    wrapper_home.joint_command.joint4 = self.home_joints_rad[3]
                    wrapper_home.joint_command.joint5 = self.home_joints_rad[4]
                    wrapper_home.joint_command.joint6 = self.home_joints_rad[5]
                    wrapper_home.joint_command.joint7 = self.home_joints_rad[6]

                    # Specify the final command in this queue for re-entry purposes
                    self.finalCommand = "Joint"

                    # Append and send commands
                    queue_msg.commands.append(wrapper_gripper_open)
                    queue_msg.commands.append(wrapper_home)
                    self.command_queue_pub.publish(queue_msg)

                    # Specify that we are NOT expecting a command to be cancelled
                    self.commandCancelled = False

                    # Change state
                    self.state = FINDING_GROUND
        elif self.state == FINDING_GROUND:
            if Event == ES_COMMAND_EXECUTED:
                if self.commandCancelled:
                    # If the command was cancelled, then we can have the EE return to the home position to begin searching for the green square
                    self.get_logger().info(f'Command cancelled, and ground is saved . . . returning home to begin search for green square')

                    # Save the value of ground
                    self.ground = self.EE_pos[2]

                    # Create a CommandQueue message
                    queue_msg = CommandQueue()
                    queue_msg.header.stamp = self.get_clock().now().to_msg()

                    wrapper_home = CommandWrapper()
                    wrapper_home.command_type = "joint"
                    wrapper_home.joint_command.joint1 = self.home_joints_rad[0]
                    wrapper_home.joint_command.joint2 = self.home_joints_rad[1]
                    wrapper_home.joint_command.joint3 = self.home_joints_rad[2]
                    wrapper_home.joint_command.joint4 = self.home_joints_rad[3]
                    wrapper_home.joint_command.joint5 = self.home_joints_rad[4]
                    wrapper_home.joint_command.joint6 = self.home_joints_rad[5]
                    wrapper_home.joint_command.joint7 = self.home_joints_rad[6]

                    # Specify the final command in this queue for re-entry purposes
                    self.finalCommand = "Joint"

                    # Append and send commands
                    queue_msg.commands.append(wrapper_home)
                    self.command_queue_pub.publish(queue_msg)

                    # Change state
                    self.state = FINDING_SQUARE
                else:
                    # Otherwise, the gripper has been homed, and we can  begin searching for the ground
                    self.get_logger().info(f'Gripper has returned to home . . . Beginning search for ground')
                    # Create a CommandQueue message
                    queue_msg = CommandQueue()
                    queue_msg.header.stamp = self.get_clock().now().to_msg()

                    # Create a CommandWrapper for the pose command to move the EE down to the ground
                    wrapper_down = CommandWrapper()
                    wrapper_down.command_type = "pose"
                    wrapper_down.pose_command.x = self.EE_pos[0]
                    wrapper_down.pose_command.y = self.EE_pos[1]
                    wrapper_down.pose_command.z = 0.035 # Change this accordingly!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    wrapper_down.pose_command.qx = 1.0
                    wrapper_down.pose_command.qy = 0.0
                    wrapper_down.pose_command.qz = 0.0
                    wrapper_down.pose_command.qw = 0.0

                    # Specify the final command in this queue for re-entry purposes
                    self.finalCommand = "Pose"

                    # Append and send commands
                    queue_msg.commands.append(wrapper_down)
                    self.command_queue_pub.publish(queue_msg)

                    # Utilize the State Machine Timer to check the force sensor 100 times per second
                    self.timer_SM = self.create_timer(0.01, self.TimeoutCallback)
            elif Event == ES_TIMEOUT:
                result = self.checkCollision()
                if result == ES_COLLISION_DETECTED:
                    # Publish a message to cancel the current command
                    msg = Bool()
                    msg.data = True
                    self.cancel_command_pub.publish(msg)

                    # Stop the timer to cease force checking
                    self.CancelTimerSM()

                    # Specify that we are now expecting a command to be cancelled
                    self.commandCancelled = True

                    self.get_logger().info(f'Ground detected . . . cancelling current EE command')

                    # # Create a CommandQueue message
                    # queue_msg = CommandQueue()
                    # queue_msg.header.stamp = self.get_clock().now().to_msg()

                    # wrapper_home = CommandWrapper()
                    # wrapper_home.command_type = "joint"
                    # wrapper_home.joint_command.joint1 = self.home_joints_rad[0]
                    # wrapper_home.joint_command.joint2 = self.home_joints_rad[1]
                    # wrapper_home.joint_command.joint3 = self.home_joints_rad[2]
                    # wrapper_home.joint_command.joint4 = self.home_joints_rad[3]
                    # wrapper_home.joint_command.joint5 = self.home_joints_rad[4]
                    # wrapper_home.joint_command.joint6 = self.home_joints_rad[5]
                    # wrapper_home.joint_command.joint7 = self.home_joints_rad[6]

                    # # Specify the final command in this queue for re-entry purposes
                    # self.finalCommand = "Joint"

                    # # Append and send commands
                    # queue_msg.commands.append(wrapper_home)
                    # self.command_queue_pub.publish(queue_msg)
        elif self.state == FINDING_SQUARE:
            if Event == ES_COMMAND_EXECUTED:
                # Camera has been moved up - search for square again
                if self.goHome:
                    self.goHome = False
                    # Create a CommandQueue message
                    queue_msg = CommandQueue()
                    queue_msg.header.stamp = self.get_clock().now().to_msg()

                    wrapper_home = CommandWrapper()
                    wrapper_home.command_type = "joint"
                    wrapper_home.joint_command.joint1 = self.home_joints_rad[0]
                    wrapper_home.joint_command.joint2 = self.home_joints_rad[1]
                    wrapper_home.joint_command.joint3 = self.home_joints_rad[2]
                    wrapper_home.joint_command.joint4 = self.home_joints_rad[3]
                    wrapper_home.joint_command.joint5 = self.home_joints_rad[4]
                    wrapper_home.joint_command.joint6 = self.home_joints_rad[5]
                    wrapper_home.joint_command.joint7 = self.home_joints_rad[6]

                    # Specify the final command in this queue for re-entry purposes
                    self.finalCommand = "Joint"

                    # Append and send commands
                    queue_msg.commands.append(wrapper_home)
                    self.command_queue_pub.publish(queue_msg)
                else:
                    msg = Bool()
                    msg.data = True
                    self.scan_square_pub.publish(msg)
                    self.get_logger().info(f'Camera has been moved accordingly . . . Sending Request to find green square')
            elif Event == ES_ITEM_UNDETECTED:
                # If the green square is not in the frame of view, move the camera up by 0.1m

                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                # Create a CommandWrapper for the pose command
                wrapper_up = CommandWrapper()
                wrapper_up.command_type = "pose"

                # Populate the pose_command with the values from the pose_array
                wrapper_up.pose_command.x = self.EE_pos[0]
                wrapper_up.pose_command.y = self.EE_pos[1]
                wrapper_up.pose_command.z = self.EE_pos[2] + 0.1
                wrapper_up.pose_command.qx = 1.0
                wrapper_up.pose_command.qy = 0.0
                wrapper_up.pose_command.qz = 0.0
                wrapper_up.pose_command.qw = 0.0

                # Specify the final command in this queue for re-entry purposes
                self.finalCommand = "Pose"

                queue_msg.commands.append(wrapper_up)
                self.command_queue_pub.publish(queue_msg)

                self.get_logger().info(f'Square not found . . . Moving camera up to expand view')
            elif Event == ES_ITEM_PARTIAL:
                # If the dollar is detected, but is too close to the border, move the camera a quarter of the distance in the direction of the dollar
                # Start by finding/updating the transformation matrix from the camera frame to the base frame
                target_frame = 'link_base'
                source_frame = 'camera_color_optical_frame'
                self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                # Don't move onto the next part in the code until we've found the transformation matrix
                while not haveTransform:
                    self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                # Get the center of the dollar in the base frame
                partialPlacePoint = self.getPointInBaseFrame(self.pixel_x_square,self.pixel_y_square)

                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                # Create a CommandWrapper for the pose command
                wrapper_side = CommandWrapper()
                wrapper_side.command_type = "pose"

                # Populate the pose_command with the values from the pose_array
                wrapper_side.pose_command.x = self.EE_pos[0] + 0.25*(partialPlacePoint[0] - self.camera_wrt_EE - self.EE_pos[0])
                wrapper_side.pose_command.y = self.EE_pos[1]+ 0.25*(partialPlacePoint[1] - self.EE_pos[1])
                wrapper_side.pose_command.z = self.EE_pos[2]
                wrapper_side.pose_command.qx = 1.0
                wrapper_side.pose_command.qy = 0.0
                wrapper_side.pose_command.qz = 0.0
                wrapper_side.pose_command.qw = 0.0

                # Specify the final command in this queue for re-entry purposes
                self.finalCommand = "Pose"

                queue_msg.commands.append(wrapper_side)
                self.command_queue_pub.publish(queue_msg)

                self.get_logger().info(f'Square found, but not entirely in frame . . . Moving camera sideways in direction of square')
            elif Event == ES_ITEM_DETECTED:
                self.get_logger().info(f'Green square found . . . Now searching for dollar bill')
                # Start by finding/updating the transformation matrix from the camera frame to the base frame
                target_frame = 'link_base'
                source_frame = 'camera_color_optical_frame'
                self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                # Don't move onto the next part in the code until we've found the transformation matrix
                while not haveTransform:
                    self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                # Get the center of the square in the base frame
                self.PlacePoint = self.getPointInBaseFrame(self.pixel_x_square,self.pixel_y_square)

                # Send a Request to find the dollar
                msg = Bool()
                msg.data = True
                self.scan_dollar_pub.publish(msg)

                # Change state
                self.state = FINDING_DOLLAR
        elif self.state == FINDING_DOLLAR:
            if Event == ES_COMMAND_EXECUTED:
                # Camera has been moved up - search for dollar again
                msg = Bool()
                msg.data = True
                self.scan_dollar_pub.publish(msg)
                self.get_logger().info(f'Camera has been moved accordingly . . . Sending Request to find dollar bill')
            elif Event == ES_ITEM_UNDETECTED:
                # If the dollar is not in the frame of view, move the camera up by 0.1m

                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                # Create a CommandWrapper for the pose command
                wrapper_up = CommandWrapper()
                wrapper_up.command_type = "pose"

                # Populate the pose_command with the values from the pose_array
                wrapper_up.pose_command.x = self.EE_pos[0]
                wrapper_up.pose_command.y = self.EE_pos[1]
                wrapper_up.pose_command.z = self.EE_pos[2] + 0.1
                wrapper_up.pose_command.qx = 1.0
                wrapper_up.pose_command.qy = 0.0
                wrapper_up.pose_command.qz = 0.0
                wrapper_up.pose_command.qw = 0.0

                # Specify the final command in this queue for re-entry purposes
                self.finalCommand = "Pose"

                queue_msg.commands.append(wrapper_up)
                self.command_queue_pub.publish(queue_msg)

                # Specify that we are not above the dollar
                self.aboveDollar = False

                self.get_logger().info(f'Dollar not found . . . Moving camera up to expand view')
            elif Event == ES_ITEM_PARTIAL:
                # If the dollar is detected, but is too close to the border, move the camera a quarter of the distance in the direction of the dollar
                # Start by finding/updating the transformation matrix from the camera frame to the base frame
                target_frame = 'link_base'
                source_frame = 'camera_color_optical_frame'
                self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                # Don't move onto the next part in the code until we've found the transformation matrix
                while not haveTransform:
                    self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                # Get the center of the dollar in the base frame
                partialDollarPoint = self.getPointInBaseFrame(self.pixel_x_dollar,self.pixel_y_dollar)

                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                # Create a CommandWrapper for the pose command
                wrapper_side = CommandWrapper()
                wrapper_side.command_type = "pose"

                # Populate the pose_command with the values from the pose_array
                wrapper_side.pose_command.x = self.EE_pos[0] + 0.25*(partialDollarPoint[0] - self.camera_wrt_EE - self.EE_pos[0])
                wrapper_side.pose_command.y = self.EE_pos[1]+ 0.25*(partialDollarPoint[1] - self.EE_pos[1])
                wrapper_side.pose_command.z = self.EE_pos[2]
                wrapper_side.pose_command.qx = 1.0
                wrapper_side.pose_command.qy = 0.0
                wrapper_side.pose_command.qz = 0.0
                wrapper_side.pose_command.qw = 0.0

                # Specify the final command in this queue for re-entry purposes
                self.finalCommand = "Pose"

                queue_msg.commands.append(wrapper_side)
                self.command_queue_pub.publish(queue_msg)

                # Specify that we are not above the dollar
                self.aboveDollar = False

                self.get_logger().info(f'Dollar found, but not entirely in frame . . . Moving camera sideways in direction of dollar')
            elif Event == ES_ITEM_DETECTED:
                # Start by finding/updating the transformation matrix from the camera frame to the base frame
                target_frame = 'link_base'
                source_frame = 'camera_color_optical_frame'
                self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                # Don't move onto the next part in the code until we've found the transformation matrix
                while not haveTransform:
                    self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                # Get the center of the dollar in the base frame
                self.DollarPoint = self.getPointInBaseFrame(self.pixel_x_dollar,self.pixel_y_dollar)

                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                if self.aboveDollar:
                    # Once the camera is above the dollar, we can begin the process to grab it
                    self.get_logger().info(f'Better position measurements acquired . . . moving gripper to appropriate height to grab')

                    # Create a CommandWrapper for the pose command to move the camera above the coin
                    wrapper_item = CommandWrapper()
                    wrapper_item = CommandWrapper()
                    wrapper_item.command_type = "pose"
                    wrapper_item.pose_command.x = self.DollarPoint[0]
                    wrapper_item.pose_command.y = self.DollarPoint[1]
                    wrapper_item.pose_command.z = self.ground + 0.05 # Change this accordingly!!!!!!!!!!!!!!!!!!!!!
                    wrapper_item.pose_command.qx = 1.0
                    wrapper_item.pose_command.qy = 0.0
                    wrapper_item.pose_command.qz = 0.0
                    wrapper_item.pose_command.qw = 0.0

                    # Add the command
                    queue_msg.commands.append(wrapper_item)

                    # Change state
                    self.state = PREPPING_GRAB
                else:
                    # Otherwise, move the camera to be above the dollar
                    self.get_logger().info(f'Dollar bill found . . . moving camera over the bill to get better position measurements')

                    # Create a CommandWrapper for the pose command to move the camera above the coin
                    wrapper_above = CommandWrapper()
                    wrapper_above = CommandWrapper()
                    wrapper_above.command_type = "pose"
                    wrapper_above.pose_command.x = self.DollarPoint[0] - self.camera_wrt_EE
                    wrapper_above.pose_command.y = self.DollarPoint[1]
                    wrapper_above.pose_command.z = self.EE_pos[2]
                    wrapper_above.pose_command.qx = 1.0
                    wrapper_above.pose_command.qy = 0.0
                    wrapper_above.pose_command.qz = 0.0
                    wrapper_above.pose_command.qw = 0.0

                    # Add the command
                    queue_msg.commands.append(wrapper_above)

                    # Specify that we anticipate to be above the dollar
                    self.aboveDollar = True
                
                # Specify the final command in this queue for re-entry purposes
                self.finalCommand = "Pose"
                
                # Publish the command queue
                self.command_queue_pub.publish(queue_msg)
        elif self.state == PREPPING_GRAB:
            if Event == ES_COMMAND_EXECUTED:
                # In this state, we rotate the gripper, then perform pick and place

                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                # Create a Wrapper for a Joint Message
                wrapper_joint = CommandWrapper()
                wrapper_joint.command_type = "joint"

                # Populate the joint command accordingly
                wrapper_joint.joint_command.joint1 = self.joint_positions[0]
                wrapper_joint.joint_command.joint2 = self.joint_positions[1]
                wrapper_joint.joint_command.joint3 = self.joint_positions[2]
                wrapper_joint.joint_command.joint4 = self.joint_positions[3]
                wrapper_joint.joint_command.joint5 = self.joint_positions[4]
                wrapper_joint.joint_command.joint6 = self.joint_positions[5]
                wrapper_joint.joint_command.joint7 = self.joint_positions[6] - self.dollarAngle

                # Create a CommandWrapper for the pose command to move the camera above the coin
                Rx = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
                Rz = np.array([[np.cos(self.dollarAngle),-np.sin(self.dollarAngle),0],[np.sin(self.dollarAngle), np.cos(self.dollarAngle), 0],[0,0,1]])
                rotMat = Rz @ Rx
                rotationObject = R.from_matrix(rotMat)
                rotQuat = rotationObject.as_quat()
                wrapper_item = CommandWrapper()
                wrapper_item.command_type = "pose"
                wrapper_item.pose_command.x = self.DollarPoint[0]
                wrapper_item.pose_command.y = self.DollarPoint[1]
                wrapper_item.pose_command.z = self.ground - 0.011 # Change this accordingly!!!!!!!!!!!!!!!!!!!!!
                wrapper_item.pose_command.qx = rotQuat[0]
                wrapper_item.pose_command.qy = rotQuat[1]
                wrapper_item.pose_command.qz = rotQuat[2]
                wrapper_item.pose_command.qw = rotQuat[3]

                # Create a CommandWrapper for the gripper command to close
                wrapper_gripper_close = CommandWrapper()
                wrapper_gripper_close.command_type = "gripper"
                wrapper_gripper_close.gripper_command.gripper_position = 0.75

                # Create a CommandWrapper for the pose command to move the gripper home
                wrapper_home = CommandWrapper()
                wrapper_home.command_type = "joint"

                # Populate the joint command accordingly
                wrapper_home.joint_command.joint1 = self.home_joints_rad[0]
                wrapper_home.joint_command.joint2 = self.home_joints_rad[1]
                wrapper_home.joint_command.joint3 = self.home_joints_rad[2]
                wrapper_home.joint_command.joint4 = self.home_joints_rad[3]
                wrapper_home.joint_command.joint5 = self.home_joints_rad[4]
                wrapper_home.joint_command.joint6 = self.home_joints_rad[5]
                wrapper_home.joint_command.joint7 = self.home_joints_rad[6]

                # Create a CommandWrapper for the pose command
                wrapper_square = CommandWrapper()
                wrapper_square.command_type = "pose"

                # Populate the pose_command with the values from the pose_array
                wrapper_square.pose_command.x = self.PlacePoint[0]
                wrapper_square.pose_command.y = self.PlacePoint[1]
                wrapper_square.pose_command.z = self.ground + 0.1 # Change this accordingly!!!!!!!!!!!!!!!!!!!!!
                wrapper_square.pose_command.qx = 1.0
                wrapper_square.pose_command.qy = 0.0
                wrapper_square.pose_command.qz = 0.0
                wrapper_square.pose_command.qw = 0.0

                # Create a CommandWrapper for the gripper command to open
                wrapper_gripper_open = CommandWrapper()
                wrapper_gripper_open.command_type = "gripper"
                wrapper_gripper_open.gripper_command.gripper_position = 0.0

                # Create a CommandWrapper for the pose command
                wrapper_home = CommandWrapper()
                wrapper_home.command_type = "joint"

                # Populate the joint command accordingly
                wrapper_home.joint_command.joint1 = self.home_joints_rad[0]
                wrapper_home.joint_command.joint2 = self.home_joints_rad[1]
                wrapper_home.joint_command.joint3 = self.home_joints_rad[2]
                wrapper_home.joint_command.joint4 = self.home_joints_rad[3]
                wrapper_home.joint_command.joint5 = self.home_joints_rad[4]
                wrapper_home.joint_command.joint6 = self.home_joints_rad[5]
                wrapper_home.joint_command.joint7 = self.home_joints_rad[6]

                #queue_msg.commands.append(wrapper_joint)
                queue_msg.commands.append(wrapper_item)
                queue_msg.commands.append(wrapper_gripper_close)
                queue_msg.commands.append(wrapper_home)
                queue_msg.commands.append(wrapper_square)
                queue_msg.commands.append(wrapper_gripper_open)
                queue_msg.commands.append(wrapper_home)
                self.command_queue_pub.publish(queue_msg)

                self.get_logger().info(f'Orienting Gripper, then performing pick and place on dollar bill')
                self.state = DUMMY_STATE
        elif self.state == DUMMY_STATE:
            2
        

if __name__ == '__main__':
    rclpy.init()
    node = PickPlaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()