#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose

from std_msgs.msg import Float64, String, Bool

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from rclpy.qos import qos_profile_sensor_data, QoSProfile

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from cv_bridge import CvBridge
import time

# Frame Imports
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper

SCANNING = 1
GRABBING = 2
PLACING = 3

class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')
        
        # Publishers
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)
        self.extracted_pub = self.create_publisher(Image,"/me314_xarm/camera/image_extracted", 1)

        # Subscribers
        self.CameraIntrinicsSubscriber = self.create_subscription(CameraInfo,"/color/camera_info",self.GetCameraIntrinsics,1) # RGB Camera Intrinsics
        self.DepthCameraIntrinicsSubscriber = self.create_subscription(CameraInfo,"/aligned_depth_to_color/camera_info",self.GetDepthCameraIntrinsics,1) # Depth Camera Intrinsics
        self.camera_subscription = self.create_subscription(Image,"/color/image_raw",self.cameraRGB_callback,qos_profile=qos_profile_sensor_data) # RGB Camera
        self.camera_subscription  # prevent unused variable warning
        self.depth_camera_subscription = self.create_subscription(Image,"/aligned_depth_to_color/image_raw",self.GetDepthCV2Image,qos_profile=qos_profile_sensor_data) # Depth Camera
        self.depth_camera_subscription  # prevent unused variable warning
        self.CommandSubscriber = self.create_subscription(String,"/me314_xarm_current_command",self.GetCurrentCommand,10) # Current Command in Execution
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.GetCurrentPose, 10) # Gets Current EE Pose

        ## CV Bridge Initialization
        self.bridge = CvBridge()

        # Frame Listener Initialization
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialization of Various Variables
        self.cv_ColorImage = None
        self.cv_DepthImage = None
        self.alpha = None
        self.beta = None
        self.u0 = None
        self.v0 = None
        self.alpha_depth = None
        self.beta_depth = None
        self.u0_depth = None
        self.v0_depth = None
        self.haveIntrinsics_RBG = False
        self.haveIntrinsics_Depth = False
        self.haveColorImage = False
        self.haveDepthImage = False
        #self.haveBaseTransform = False
        self.previousCommand = ""
        self.finalCommand = None
        self.runPlanner = True
        #self.haveGrabbed = False
        self.EE_pos = None
        self.foundBlockPoint = False
        self.foundPlacePoint = False # Set True if using Gazebo without Green Square
        self.blockPoint = None
        self.PlacePoint = None#np.array([0.35, -0.1, 0.075]) # this is filled in for the case where Gazebo does not have the green square
        self.state = SCANNING

        self.timer = self.create_timer(3.0, self.Planner)

    def Planner(self):
        if self.runPlanner and self.haveColorImage and self.haveDepthImage and self.haveIntrinsics_Depth and self.haveIntrinsics_RBG:
            # Start by finding/updating the transformation matrix from the camera frame to the base frame
            target_frame = 'link_base'
            source_frame = 'camera_color_optical_frame'
            self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
            if haveTransform:
                self.runPlanner = False # The Planner is only designed to run after xarm_commander has finished command sequences - comment out when testing green detection
                #self.haveBaseTransform = True
                if self.state == SCANNING:
                    # Look for the red cube if not already found
                    if not self.foundBlockPoint:
                        self.blockPoint, self.foundBlockPoint = self.findCenterRedPixel()
                    # Look for the green square if not already found
                    if not self.foundPlacePoint:
                        self.PlacePoint, self.foundPlacePoint = self.findCenterGreenPixel()
                    if self.foundBlockPoint and self.foundPlacePoint:
                        # If both have been found, then we can move to the GRABBING state.
                        # But first, we need to ensure that the EE is back at the HOME state and that the gripper is open

                        # Create a CommandQueue message
                        queue_msg = CommandQueue()
                        queue_msg.header.stamp = self.get_clock().now().to_msg()

                        # Create a CommandWrapper for the gripper command to open
                        wrapper_gripper_open = CommandWrapper()
                        wrapper_gripper_open.command_type = "gripper"
                        wrapper_gripper_open.gripper_command.gripper_position = 0.0

                        # Create a CommandWrapper for the pose command
                        wrapper_home = CommandWrapper()
                        wrapper_home.command_type = "pose"

                        # Populate the pose_command with the values from the pose_array
                        wrapper_home.pose_command.x = 0.3408
                        wrapper_home.pose_command.y = 0.0021
                        wrapper_home.pose_command.z = 0.3029
                        wrapper_home.pose_command.qx = 1.0
                        wrapper_home.pose_command.qy = 0.0
                        wrapper_home.pose_command.qz = 0.0
                        wrapper_home.pose_command.qw = 0.0

                        # Specify the final command in this queue for re-entry purposes
                        self.finalCommand = "Pose"

                        queue_msg.commands.append(wrapper_gripper_open)
                        queue_msg.commands.append(wrapper_home)
                        self.command_queue_pub.publish(queue_msg)

                        self.get_logger().info(f'Red Cube and Green Square Found.  Returning to home pose first before executing Pick and Place')

                        self.state = GRABBING
                    else:
                        # If we haven't found both items, we raise the camera higher to expand the field of view

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

                        self.get_logger().info(f'Moving camera up to expand view')
                elif self.state == GRABBING:
                    # Create a CommandQueue message
                    queue_msg = CommandQueue()
                    queue_msg.header.stamp = self.get_clock().now().to_msg()
                    
                    # Create a CommandWrapper for the pose command
                    wrapper_p1 = CommandWrapper()
                    wrapper_p1.command_type = "pose"

                    # Populate the pose_command with the values from the pose_array
                    wrapper_p1.pose_command.x = self.blockPoint[0]
                    wrapper_p1.pose_command.y = self.blockPoint[1]
                    wrapper_p1.pose_command.z = self.blockPoint[2] - 0.01
                    wrapper_p1.pose_command.qx = 1.0
                    wrapper_p1.pose_command.qy = 0.0
                    wrapper_p1.pose_command.qz = 0.0
                    wrapper_p1.pose_command.qw = 0.0

                    # Create a CommandWrapper for the gripper command to close
                    wrapper_gripper_close = CommandWrapper()
                    wrapper_gripper_close.command_type = "gripper"
                    wrapper_gripper_close.gripper_command.gripper_position = 1.0

                    # Create a CommandWrapper for the pose command
                    wrapper_home = CommandWrapper()
                    wrapper_home.command_type = "pose"

                    # Populate the pose_command with the values from the pose_array
                    wrapper_home.pose_command.x = 0.3408
                    wrapper_home.pose_command.y = 0.0021
                    wrapper_home.pose_command.z = 0.3029
                    wrapper_home.pose_command.qx = 1.0
                    wrapper_home.pose_command.qy = 0.0
                    wrapper_home.pose_command.qz = 0.0
                    wrapper_home.pose_command.qw = 0.0

                    # Specify the final command in this queue for re-entry purposes
                    self.finalCommand = "Pose"

                    # Add the command to the queue and publish
                    queue_msg.commands.append(wrapper_p1)
                    queue_msg.commands.append(wrapper_gripper_close)
                    queue_msg.commands.append(wrapper_home)
                    self.command_queue_pub.publish(queue_msg)

                    self.get_logger().info(f'Commands for picking sent')

                    self.state = PLACING
                elif self.state == PLACING:
                    # Create a CommandQueue message
                    queue_msg = CommandQueue()
                    queue_msg.header.stamp = self.get_clock().now().to_msg()
                    
                    # Create a CommandWrapper for the pose command
                    wrapper_p2 = CommandWrapper()
                    wrapper_p2.command_type = "pose"

                    # Populate the pose_command with the values from the pose_array
                    wrapper_p2.pose_command.x = self.PlacePoint[0]
                    wrapper_p2.pose_command.y = self.PlacePoint[1]
                    wrapper_p2.pose_command.z = self.PlacePoint[2]
                    wrapper_p2.pose_command.qx = 1.0
                    wrapper_p2.pose_command.qy = 0.0
                    wrapper_p2.pose_command.qz = 0.0
                    wrapper_p2.pose_command.qw = 0.0

                    # Create a CommandWrapper for the gripper command to open
                    wrapper_gripper_open = CommandWrapper()
                    wrapper_gripper_open.command_type = "gripper"
                    wrapper_gripper_open.gripper_command.gripper_position = 0.0

                    # Create a CommandWrapper for the pose command
                    wrapper_home = CommandWrapper()
                    wrapper_home.command_type = "pose"

                    # Populate the pose_command with the values from the pose_array
                    wrapper_home.pose_command.x = 0.3408
                    wrapper_home.pose_command.y = 0.0021
                    wrapper_home.pose_command.z = 0.3029
                    wrapper_home.pose_command.qx = 1.0
                    wrapper_home.pose_command.qy = 0.0
                    wrapper_home.pose_command.qz = 0.0
                    wrapper_home.pose_command.qw = 0.0

                    # Specify the final command in this queue for re-entry purposes
                    self.finalCommand = "Pose"

                    # Add the command to the queue and publish
                    queue_msg.commands.append(wrapper_p2)
                    queue_msg.commands.append(wrapper_gripper_open)
                    queue_msg.commands.append(wrapper_home)
                    self.command_queue_pub.publish(queue_msg)

                    self.get_logger().info(f'Commands for placing sent')

                    self.timer.cancel()
    
    def GetTransform(self, target_frame, source_frame):
        try:
            #now = self.get_clock().now()
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(seconds=0)
            )
            #self.get_logger().info(f'{transform}')
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
    
    def GetDepthCameraIntrinsics(self, DepthCameraMsg):
        self.get_logger().info(f'Getting depth camera intrinsics')
        self.alpha_depth = DepthCameraMsg.k[0]
        self.beta_depth = DepthCameraMsg.k[4]
        self.u0_depth = DepthCameraMsg.k[2]
        self.v0_depth = DepthCameraMsg.k[5]
        self.destroy_subscription(self.DepthCameraIntrinicsSubscriber)
        self.haveIntrinsics_Depth = True
    
    def GetDepthCV2Image(self,DepthImageMsg):
        self.cv_DepthImage = self.bridge.imgmsg_to_cv2(DepthImageMsg, desired_encoding='passthrough')
        self.haveDepthImage = True
    
    def cameraRGB_callback(self, ImageMsg: Image):
        self.cv_ColorImage = self.bridge.imgmsg_to_cv2(ImageMsg, "rgb8")
        self.haveColorImage = True
    
    def findCenterRedPixel(self):
        hsv_img = cv2.cvtColor(self.cv_ColorImage,cv2.COLOR_RGB2HSV)
        lower_bound = np.array([0, 100, 90])
        upper_bound = np.array([10, 255, 255])
        mask_img = cv2.inRange(hsv_img, lower_bound, upper_bound)
        modified_img = cv2.bitwise_and(self.cv_ColorImage, self.cv_ColorImage, mask=mask_img)
        self.extracted_pub.publish(self.bridge.cv2_to_imgmsg(modified_img, "rgb8"))
        cube_points = [[u,v] for u in range(mask_img.shape[0]) for v in range(mask_img.shape[1]) if mask_img[u,v]>0 ]
        if len(cube_points) != 0:
            cube_center = np.floor(np.mean(cube_points,0))
            u = int(cube_center[1])
            v = int(cube_center[0])
            return self.getPointInBaseFrame(u,v), True
        else:
            self.get_logger().info(f'Cube not found')
            return None, False
    
    def findCenterGreenPixel(self):
        hsv_img = cv2.cvtColor(self.cv_ColorImage,cv2.COLOR_RGB2HSV)
        lower_bound = np.array([40, 50, 0])
        upper_bound = np.array([80, 255, 255])
        mask_img = cv2.inRange(hsv_img, lower_bound, upper_bound)
        modified_img = cv2.bitwise_and(self.cv_ColorImage, self.cv_ColorImage, mask=mask_img)
        self.extracted_pub.publish(self.bridge.cv2_to_imgmsg(modified_img, "rgb8"))
        cube_points = [[u,v] for u in range(mask_img.shape[0]) for v in range(mask_img.shape[1]) if mask_img[u,v]>0 ]
        if len(cube_points) != 0:
            cube_center = np.floor(np.mean(cube_points,0))
            u = int(cube_center[1])
            v = int(cube_center[0])
            return self.getPointInBaseFrame(u,v), True
        else:
            self.get_logger().info(f'Green square not found')
            return None, False
    
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
    
    def GetCurrentCommand(self,command):
        #self.get_logger().info(f'Command = {command.data}')
        if command.data != self.previousCommand:
            if self.finalCommand in self.previousCommand and command.data == "":
                self.runPlanner = True
        self.previousCommand = command.data
    
    def GetCurrentPose(self, msg: Pose):
        self.EE_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        

if __name__ == '__main__':
    rclpy.init()
    node = PickPlaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()