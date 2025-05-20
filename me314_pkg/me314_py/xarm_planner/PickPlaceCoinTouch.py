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
INITIALIZATION = 1
FINDING_SQUARE = 2
FINDING_ITEM = 3
PREPPING_GRAB = 4
DUMMY_STATE = 5

# Events
ES_TIMEOUT = 11
ES_COMMAND_EXECUTED = 12
ES_ITEM_DETECTED = 13
ES_ITEM_UNDETECTED = 14
ES_ITEM_PARTIAL = 15
ES_COLLISION_DETECTED = 16
ES_NO_COLLISION_DETECTED = 17

class PickPlaceCoinNode(Node):
    def __init__(self):
        super().__init__('pick_place_coin_node')

        # Use this to set the name of the camera topics depending on whether we're in sim or real
        self.usingRealRobot = True

        # Camera Topic Names
        self.CameraIntrinsicsTopic = "/color/camera_info"
        self.CameraTopic = "/color/image_raw"
        self.DepthCameraTopic = "/aligned_depth_to_color/image_raw"
        if self.usingRealRobot:
            self.CameraIntrinsicsTopic = "/camera/realsense2_camera_node" + self.CameraIntrinsicsTopic
            self.CameraTopic = "/camera/realsense2_camera_node" + self.CameraTopic
            self.DepthCameraTopic = "/camera/realsense2_camera_node" + self.DepthCameraTopic
        
        # Publishers
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)
        self.scan_coin_pub = self.create_publisher(Bool, '/scan_coin_request', 10)
        self.cancel_command_pub = self.create_publisher(Bool, '/me314_xarm_cancel_command', 10)

        # Subscribers
        self.CameraIntrinicsSubscriber = self.create_subscription(CameraInfo,self.CameraIntrinsicsTopic,self.GetCameraIntrinsics,1) # RGB Camera Intrinsics
        self.camera_subscription = self.create_subscription(Image,self.CameraTopic,self.cameraRGB_callback,qos_profile=qos_profile_sensor_data) # RGB Camera
        self.camera_subscription  # prevent unused variable warning
        self.depth_camera_subscription = self.create_subscription(Image,self.DepthCameraTopic,self.GetDepthCV2Image,qos_profile=qos_profile_sensor_data) # Depth Camera
        self.depth_camera_subscription  # prevent unused variable warning
        self.CommandSubscriber = self.create_subscription(String,"/me314_xarm_current_command",self.GetCurrentCommand,10) # Current Command in Execution
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.GetCurrentPose, 10) # Gets Current EE Pose
        self.JointAngleSubscriber = self.create_subscription(JointState, '/me314_xarm_current_joint_positions_deg', self.GetJointAngles, 10) # Gets current joint angles
        self.coin_sub = self.create_subscription(Point, "/coin_report", self.CoinCallback, 10) # Gets report regarding coin
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
        self.haveColorImage = False
        self.haveDepthImage = False
        self.previousCommand = ""
        self.finalCommand = None
        self.baseTransform = None
        self.EE_pos = None
        self.aboveItem = False
        self.ItemPoint = None
        self.PlacePoint = None
        self.haveWrench = False
        self.currentForce = None
        self.ground = None
        self.FirstTimeout = False
        self.state = INITIALIZATION

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
    
    def cameraRGB_callback(self, ImageMsg: Image):
        self.cv_ColorImage = self.bridge.imgmsg_to_cv2(ImageMsg, "rgb8")
        self.haveColorImage = True
    
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
        elif point_base[2] < 0.015:
            point_base[2] = 0.015
        self.get_logger().info(f'Desired object Base location: x={point_base[0]:.2f}, y={point_base[1]:.2f}, z={point_base[2]:.2f}')
        return point_base
    
    def GetCurrentPose(self, msg: Pose):
        self.EE_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
    
    def GetJointAngles(self,JointMessage):
        self.joint_positions = [math.radians(pos) for pos in JointMessage.position]
    
    def GetCurrentCommand(self,command):
        #self.get_logger().info(f'Command = {command.data}')
        if command.data != self.previousCommand:
            if self.finalCommand in self.previousCommand and command.data == "":
                self.StateMachine(ES_COMMAND_EXECUTED)
        self.previousCommand = command.data
    
    def CoinCallback(self,msg):
        if msg.z == 1.0:
            self.pixel_x_item = msg.x
            self.pixel_y_item = msg.y
            self.StateMachine(ES_ITEM_DETECTED)
        else:
            self.StateMachine(ES_ITEM_UNDETECTED)
    
    def findCenterGreenPixel(self):
        hsv_img = cv2.cvtColor(self.cv_ColorImage,cv2.COLOR_RGB2HSV)
        lower_bound = np.array([40, 50, 0])
        upper_bound = np.array([80, 255, 255])
        mask_img = cv2.inRange(hsv_img, lower_bound, upper_bound)
        # modified_img = cv2.bitwise_and(self.cv_ColorImage, self.cv_ColorImage, mask=mask_img)
        # self.extracted_pub.publish(self.bridge.cv2_to_imgmsg(modified_img, "rgb8"))
        verticalSize = mask_img.shape[0]
        horizontalSize = mask_img.shape[1]
        cube_points = [[y,x] for y in range(verticalSize) for x in range(horizontalSize) if mask_img[y,x]>0 ]
        if len(cube_points) != 0:
            cube_center = np.floor(np.mean(cube_points,0))
            self.pixel_x_square = int(cube_center[1])
            self.pixel_y_square = int(cube_center[0])
            if (self.pixel_x_square > 0.2 * horizontalSize and self.pixel_x_square < 0.8 * horizontalSize
            and self.pixel_y_square > 0.2 * verticalSize and self.pixel_y_square < 0.8 * verticalSize):
                return ES_ITEM_DETECTED
            else:
                return ES_ITEM_PARTIAL
        else:
            return ES_ITEM_UNDETECTED
    
    def ForceCallback(self,msg):
        self.currentForce = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        #self.currentTorque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
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
        if baseForce[2] > 9:
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
                if self.haveDepthImage and self.haveIntrinsics_RBG and self.haveColorImage and self.haveWrench:
                    # Cancel the Timer that continually checks if we have the stuff
                    self.CancelTimerSM()
                    
                    self.get_logger().info(f'Camera and force stuff has been obtained . . . Sending Command to ensure gripper is in home position')
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

                    queue_msg.commands.append(wrapper_home)
                    self.command_queue_pub.publish(queue_msg)

                    # Change state
                    self.state = FINDING_SQUARE
        elif self.state == FINDING_SQUARE:
            if Event == ES_TIMEOUT or Event == ES_COMMAND_EXECUTED:
                if Event == ES_TIMEOUT:
                    # Stop the timer, as the State Machine is able to continue running on its own in this state
                    self.CancelTimerSM()
                    self.get_logger().info(f'Looking for the green square')
                else:
                    # Camera has been moved up - search for square again
                    self.get_logger().info(f'Camera has been moved accordingly . . . Sending Request to find green square')
                # Search for the center of the green square
                result = self.findCenterGreenPixel()
                # Act accordingly
                if result == ES_ITEM_UNDETECTED:
                    # If the green square is not in the frame of view, move the camera up by 0.1m

                    # Create a CommandQueue message
                    queue_msg = CommandQueue()
                    queue_msg.header.stamp = self.get_clock().now().to_msg()

                    # Create a CommandWrapper for the pose command
                    wrapper_up = CommandWrapper()
                    wrapper_up.command_type = "pose"
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
                elif result == ES_ITEM_PARTIAL:
                    # If the square is detected, but is too close to the border, move the camera a quarter of the distance in the direction of the square
                    # Start by finding/updating the transformation matrix from the camera frame to the base frame
                    target_frame = 'link_base'
                    source_frame = 'camera_color_optical_frame'
                    self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                    # Don't move onto the next part in the code until we've found the transformation matrix
                    while not haveTransform:
                        self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                    # Get the center of the partial square in the base frame
                    partialSquarePoint = self.getPointInBaseFrame(self.pixel_x_square,self.pixel_y_square)

                    # Create a CommandQueue message
                    queue_msg = CommandQueue()
                    queue_msg.header.stamp = self.get_clock().now().to_msg()

                    # Create a command for moving EE to the side
                    wrapper_side = CommandWrapper()
                    wrapper_side.command_type = "pose"
                    wrapper_side.pose_command.x = self.EE_pos[0] + 0.1*(partialSquarePoint[0] - self.camera_wrt_EE - self.EE_pos[0])
                    wrapper_side.pose_command.y = self.EE_pos[1]+ 0.1*(partialSquarePoint[1] - self.EE_pos[1])
                    wrapper_side.pose_command.z = self.EE_pos[2]
                    wrapper_side.pose_command.qx = 1.0
                    wrapper_side.pose_command.qy = 0.0
                    wrapper_side.pose_command.qz = 0.0
                    wrapper_side.pose_command.qw = 0.0

                    # Specify the final command in this queue for re-entry purposes
                    self.finalCommand = "Pose"

                    queue_msg.commands.append(wrapper_side)
                    self.command_queue_pub.publish(queue_msg)

                    self.get_logger().info(f'Green square found, but not entirely in frame . . . Moving camera sideways in direction of square')
                elif result == ES_ITEM_DETECTED:
                    self.get_logger().info(f'Green square found . . . Beginning to search for coin')
                    # Start by finding/updating the transformation matrix from the camera frame to the base frame
                    target_frame = 'link_base'
                    source_frame = 'camera_color_optical_frame'
                    self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                    # Don't move onto the next part in the code until we've found the transformation matrix
                    while not haveTransform:
                        self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                    # Get the center of the square in the base frame
                    self.PlacePoint = self.getPointInBaseFrame(self.pixel_x_square,self.pixel_y_square)

                    # Begin to search for coin
                    msg = Bool()
                    msg.data = True
                    self.scan_coin_pub.publish(msg)

                    # Change state
                    self.state = FINDING_ITEM
        elif self.state == FINDING_ITEM:
            if Event == ES_COMMAND_EXECUTED:
                # Camera has been moved up - search for coin again
                msg = Bool()
                msg.data = True
                self.scan_coin_pub.publish(msg)
                self.get_logger().info(f'Camera has been moved accordingly . . . Sending Request to find coin')
            elif Event == ES_ITEM_UNDETECTED:
                # If the coin is not in the frame of view, move the camera up by 0.1m

                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                # Create a CommandWrapper for the pose command
                wrapper_up = CommandWrapper()
                wrapper_up.command_type = "pose"
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

                # Ensure that we know that the camera is not above the item
                self.aboveItem = False

                self.get_logger().info(f'Coin not found . . . Moving camera up to expand view')
            elif Event == ES_ITEM_DETECTED:
                # Start by finding/updating the transformation matrix from the camera frame to the base frame
                target_frame = 'link_base'
                source_frame = 'camera_color_optical_frame'
                self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                # Don't move onto the next part in the code until we've found the transformation matrix
                while not haveTransform:
                    self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
                # Get the center of the dollar in the base frame
                self.ItemPoint = self.getPointInBaseFrame(self.pixel_x_item,self.pixel_y_item)

                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                if self.aboveItem:
                    # If we are above the item, we can start the sequence for finding the ground's height
                    self.get_logger().info(f'Camera above the coin . . . Now beginning routine to find the height of the ground')

                    # Create a CommandWrapper for the pose command to move the EE above the coin
                    wrapper_item = CommandWrapper()
                    wrapper_item.command_type = "pose"
                    wrapper_item.pose_command.x = self.ItemPoint[0]
                    wrapper_item.pose_command.y = self.ItemPoint[1]
                    wrapper_item.pose_command.z = self.EE_pos[2]
                    wrapper_item.pose_command.qx = 1.0
                    wrapper_item.pose_command.qy = 0.0
                    wrapper_item.pose_command.qz = 0.0
                    wrapper_item.pose_command.qw = 0.0

                    # Create a CommandWrapper for the gripper command to open
                    wrapper_gripper_open = CommandWrapper()
                    wrapper_gripper_open.command_type = "gripper"
                    wrapper_gripper_open.gripper_command.gripper_position = 0.0 # Change this accordingly!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    # Create a CommandWrapper for the pose command to move the EE down to the ground
                    wrapper_down = CommandWrapper()
                    wrapper_down.command_type = "pose"
                    wrapper_down.pose_command.x = self.ItemPoint[0]
                    wrapper_down.pose_command.y = self.ItemPoint[1]
                    wrapper_down.pose_command.z = 0.022 # Change this accordingly!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    wrapper_down.pose_command.qx = 1.0
                    wrapper_down.pose_command.qy = 0.0
                    wrapper_down.pose_command.qz = 0.0
                    wrapper_down.pose_command.qw = 0.0

                    # Add the commands
                    queue_msg.commands.append(wrapper_item)
                    queue_msg.commands.append(wrapper_gripper_open)
                    queue_msg.commands.append(wrapper_down)

                    # Change state
                    self.state = PREPPING_GRAB

                    # Utilize the State Machine Timer to check the force sensor 100 times per second in the next state
                    self.timer_SM = self.create_timer(0.01, self.TimeoutCallback)
                else:
                    # Otherwise, we move the camera so that the item is right beneath it
                    self.get_logger().info(f'Coin found . . . Moving camera to be right above the coin for better location measurements')

                    # Create a CommandWrapper for the pose command to move the camera above the coin
                    wrapper_above = CommandWrapper()
                    wrapper_above = CommandWrapper()
                    wrapper_above.command_type = "pose"
                    wrapper_above.pose_command.x = self.ItemPoint[0] - self.camera_wrt_EE
                    wrapper_above.pose_command.y = self.ItemPoint[1]
                    wrapper_above.pose_command.z = self.EE_pos[2]
                    wrapper_above.pose_command.qx = 1.0
                    wrapper_above.pose_command.qy = 0.0
                    wrapper_above.pose_command.qz = 0.0
                    wrapper_above.pose_command.qw = 0.0

                    # Add the command
                    queue_msg.commands.append(wrapper_above)

                    # Specify that we expect to be above the coin the next time we enter this state
                    self.aboveItem = True
                
                # Specify the final command in this queue for re-entry purposes
                self.finalCommand = "Pose"
                
                # Publish the command queue
                self.command_queue_pub.publish(queue_msg)
        elif self.state == PREPPING_GRAB:
            if Event == ES_TIMEOUT:
                result = self.checkCollision()
                if result == ES_COLLISION_DETECTED:
                    # Publish a message to cancel the current command
                    msg = Bool()
                    msg.data = True
                    self.cancel_command_pub.publish(msg)

                    # Stop the timer to cease force checking
                    self.CancelTimerSM()

                    self.get_logger().info(f'Ground detected . . . cancelling current EE command')
            elif Event == ES_COMMAND_EXECUTED:
                # Once the downward command is cancelled, we can save the value of the ground, move the EE up, then perform pick and place

                self.ground = self.EE_pos[2]

                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                # Create a CommandWrapper for the pose command to move the EE down to the ground
                wrapper_shift_up = CommandWrapper()
                wrapper_shift_up.command_type = "pose"
                wrapper_shift_up.pose_command.x = self.ItemPoint[0]
                wrapper_shift_up.pose_command.y = self.ItemPoint[1]
                wrapper_shift_up.pose_command.z = self.ground + 0.008    # Change this accordingly!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                wrapper_shift_up.pose_command.qx = 1.0
                wrapper_shift_up.pose_command.qy = 0.0
                wrapper_shift_up.pose_command.qz = 0.0
                wrapper_shift_up.pose_command.qw = 0.0

                # Create a CommandWrapper for the gripper command to close
                wrapper_gripper_close = CommandWrapper()
                wrapper_gripper_close.command_type = "gripper"
                wrapper_gripper_close.gripper_command.gripper_position = 0.75 # Change this accordingly!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                # Create a CommandWrapper for the pose command to move the gripper home
                wrapper_home = CommandWrapper()
                wrapper_home.command_type = "joint"
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
                wrapper_square.pose_command.x = self.PlacePoint[0]
                wrapper_square.pose_command.y = self.PlacePoint[1]
                wrapper_square.pose_command.z = self.ground + 0.03 # Change this accordingly!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                wrapper_square.pose_command.qx = 1.0
                wrapper_square.pose_command.qy = 0.0
                wrapper_square.pose_command.qz = 0.0
                wrapper_square.pose_command.qw = 0.0

                # Create a CommandWrapper for the gripper command to open
                wrapper_gripper_open = CommandWrapper()
                wrapper_gripper_open.command_type = "gripper"
                wrapper_gripper_open.gripper_command.gripper_position = 0.0

                # Publish the commands
                queue_msg.commands.append(wrapper_shift_up)
                queue_msg.commands.append(wrapper_gripper_close)
                queue_msg.commands.append(wrapper_home)
                queue_msg.commands.append(wrapper_square)
                queue_msg.commands.append(wrapper_gripper_open)
                queue_msg.commands.append(wrapper_home)
                self.command_queue_pub.publish(queue_msg)

                self.get_logger().info(f'Performing pick and place on coin')
                self.state = DUMMY_STATE
        elif self.state == DUMMY_STATE:
            2
        

if __name__ == '__main__':
    rclpy.init()
    node = PickPlaceCoinNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()