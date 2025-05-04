#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point

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
import torch
import os
from PIL import Image as PIL_Image
from ament_index_python.packages import get_package_share_directory
import argparse

# Frame Imports
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper
from sensor_msgs.msg import JointState

class CoinFinderNode(Node):
    def __init__(self, visualize=False, real=False):
        super().__init__('pick_place_node')

         # Get the package share directory to be used to find model weights
        pkg_path = get_package_share_directory('me314_pkg')
        
        # Process parser arguments
        self.real = real
        self.visualize = visualize

        # Publishers
        # Publisher for coin pose
        self.coin_point_publisher = self.create_publisher(
            Point,  # Change to the required message type
            '/coin_report', 
            10
        )

        # Subscribers
        # Create a subscriber to get the image from the realsense
        if self.real:
            self.camera_subscription = self.create_subscription(Image,"/camera/realsense2_camera_node/color/image_raw",self.CameraRGBCallback,qos_profile=qos_profile_sensor_data) # RGB Camera
        else:
            self.camera_subscription = self.create_subscription(Image,"/color/image_raw",self.CameraRGBCallback,qos_profile=qos_profile_sensor_data) # RGB Camera
        
        self.scan_image_for_square_subscription = self.create_subscription(
            Bool,  # Message type
            '/scan_coin_request',  # Topic name
            self.ListenerCallbackGetCoin,  # Callback function
            10,  # Queue size
        )

        ## CV Bridge Initialization
        self.bridge = CvBridge()

        # Initialization of Various Variables
        self.cv_image = None # used to store the rgm image from camera


        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 25
        params.maxThreshold = 220
        params.thresholdStep = 10

        # Increase minimum area
        params.filterByArea = True
        params.minArea = 400 
        
        # Filter by circularity 
        params.filterByCircularity = True
        params.minCircularity = 0.7

        # Disable unnecessary filters.
        params.filterByConvexity = False
        #params.filterByInertia = False

        self.blob_detector = cv2.SimpleBlobDetector_create(params)


    def CameraRGBCallback(self, ImageMsg: Image):
        self.cv_image = self.bridge.imgmsg_to_cv2(ImageMsg, "rgb8")
        self.haveColorImage = True

    def ListenerCallbackGetSquare(self, msg):
        """
        Callback function that is triggered when this node is told to look for coin.
        """
        if msg.data:  # Check if the boolean value is True
            self.get_logger().info('Received True! Running function to find coin...')
            
            self.getObjectPose("Coin")
    

    def getObjectPose(self, object):
        coin_point = Point()
        # Case to look for dollar
        
        if object == "Coin":
            print("Looking for Coin")

            frame = self.cv_image.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            keypoints = self.blob_detector.detect(thresh)

            # Draw detected blobs as green circles.
            output = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 255, 0),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                
            if len(keypoints) == 0:
                print("No Coins not found")
                coin_point.z = 0.0   

            else:
                print(len(keypoints), "Coin(s) found")
                coin_point.x = float(keypoints[0].pt[0])
                coin_point.y = float(keypoints[0].pt[1])
                coin_point.z = 1.0
                         
            self.coin_point_publisher.publish(coin_point)

        if self.visualize:
            cv2.imshow("Visualization", output)
        cv2.waitKey(0)

def main(args=None):
    parser = argparse.ArgumentParser()    
    parser.add_argument('-v', '--visualize', 
                       action='store_true',
                       help="Enable visualization")
    parser.add_argument('-r', '--real', 
                       action='store_true',
                       help="Change topics to use topics from real robot")
    # Separate known args from ROS args
    known_args, remaining_args = parser.parse_known_args(args=args)
    # Initialize ROS with remaining args
    rclpy.init(args=remaining_args)
    node = CoinFinderNode(
        visualize=known_args.visualize,
        real=known_args.real)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()