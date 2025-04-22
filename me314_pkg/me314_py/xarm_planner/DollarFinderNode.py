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

# Frame Imports
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper
from sensor_msgs.msg import JointState

class DollarFinderNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')
        
        # Publishers
        # Publisher for dollar pose
        self.dollar_pose_publisher = self.create_publisher(
            Pose,  # Change to the required message type
            '/dollar_report', 
            10
        )
        # Publisher for square pose
        self.square_point_publisher = self.create_publisher(
            Point,  # Change to the required message type
            '/square_report', 
            10
        )

        # Subscribers
        # Create a subscriber to get the image from the realsense
        self.camera_subscription = self.create_subscription(Image,"/color/image_raw",self.CameraRGBCallback,qos_profile=qos_profile_sensor_data) # RGB Camera

        
        #self.camera_subscription = self.create_subscription(Image,"/camera/realsense2_camera_node/color/image_raw",self.CameraRGBCallback,qos_profile=qos_profile_sensor_data) # RGB Camera
        
        # Create a subscriber to boolean topic that tells this node to run
        self.scan_image_for_dollar_subscription = self.create_subscription(
            Bool,  # Message type
            '/scan_dollar_request',  # Topic name
            self.ListenerCallbackGetDollar,  # Callback function
            10,  # Queue size
        )
        self.scan_image_for_square_subscription = self.create_subscription(
            Bool,  # Message type
            '/scan_square_request',  # Topic name
            self.ListenerCallbackGetSquare,  # Callback function
            10,  # Queue size
        )

        ## CV Bridge Initialization
        self.bridge = CvBridge()

        # Initialization of Various Variables
        self.cv_image = None # used to store the rgm image from camera

    def CameraRGBCallback(self, ImageMsg: Image):
        self.cv_image = self.bridge.imgmsg_to_cv2(ImageMsg, "rgb8")
        self.haveColorImage = True

    def ListenerCallbackGetDollar(self, msg):
        """
        Callback function that is triggered when this node is told to look for dollar.
        """
        if msg.data:  # Check if the boolean value is True
            self.get_logger().info('Received True! Running function to find dollar...')
            
            self.getObjectPose('Dollar')

    def ListenerCallbackGetSquare(self, msg):
        """
        Callback function that is triggered when this node is told to look for square.
        """
        if msg.data:  # Check if the boolean value is True
            self.get_logger().info('Received True! Running function to find square...')
            
            self.getObjectPose('Square')
    
    def angle_between_vectors(self,a, b):
        """Compute angle (in degrees) between two vectors a and b."""
        dot = np.dot(a, b)
        det = np.cross(a, b)  # Determinant (a_x*b_y - a_y*b_x)
        angle_rad = np.arctan2(det, dot) 
        
        return np.degrees(angle_rad)

    def getObjectPose(self, object):
        hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up the mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        quad_info = []  # Store (x, y, w, h) for each quad
        
  
        dollar_pose = Pose()
        square_point = Point()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Skip small detections
                continue
            
            # Approximate the contour to a polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            if len(approx) == 4:
                # Get the rotated rectangle (non-axis-aligned)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                
                # Draw the rotated rectangle
                cv2.drawContours(self.cv_image, [box], 0, (0,255,255), 2)
                
                # Calculate center point
                center_x = int(rect[0][0])
                center_y = int(rect[0][1])
                center = (center_x, center_y)
                
                # Draw center point
                cv2.circle(self.cv_image, center, 5, (255, 0, 255), -1)
                
                # Get width and height (sorted so w > h)
                w, h = sorted([rect[1][0], rect[1][1]], reverse=True)
                aspect_ratio = w/h

                if object == "Dollar":
                    if aspect_ratio >= 1.5:
                        print("Dollar found")
                        # Find the bottom left point
                        sorted_by_y = sorted(box, key=lambda p: -p[1])  # Sort by Y descending
                        # Compare the distances of each point to the bottom left point
                        dist = np.linalg.norm(box - sorted_by_y[0], axis=1)
                        # Get the distances of the two closes points
                        closest_points_index = np.argsort(dist)[1:3]
                        # Create two vectors from the bottom left point of the rectangle to the closest two points
                        vector1 = box[closest_points_index[0]] - sorted_by_y[0]
                        vector2 = box[closest_points_index[1]] - sorted_by_y[0]
                        if np.linalg.norm(vector1) < np.linalg.norm(vector2):
                            vector = vector1
                        else:
                            vector = vector2
                        angle = self.angle_between_vectors(vector, np.array([1,0]))
                        print(angle)
                        cv2.putText(self.cv_image, f"Dollar, angle: {angle:.1f}°", tuple(box[0]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        
                        #pose = Pose()
                        dollar_pose.position.x = float(center_x)
                        dollar_pose.position.y = float(center_y)
                        dollar_pose.position.z = 1.0
                        dollar_pose.orientation.z = angle
                        #self.dollar_pose_publisher.publish(pose)
                        break

                elif object == "Square":
                    if aspect_ratio < 1.5:
                        print("Square found")
                        cv2.putText(self.cv_image, "Square", tuple(box[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                        
                        #point = Point()
                        square_point.x = float(center_x)
                        square_point.y = float(center_y)
                        square_point.z = 1.0
                        #self.square_point_publisher.publish(point)
                        break
            else:
                if object == "Dollar":
                    print("Dollar not found")
                    #pose = Pose()
                    dollar_pose.position.z = 0.0
                    #self.dollar_pose_publisher.publish(pose)
                elif object == "Square":
                    print("Square not found")
                    #point = Point()
                    square_point.z = 0.0
                    #self.square_point_publisher.publish(point)

        if object == 'Dollar':
            self.dollar_pose_publisher.publish(dollar_pose)
        elif object == 'Square':
            self.square_point_publisher.publish(square_point)


if __name__ == '__main__':
    rclpy.init()
    node = DollarFinderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()