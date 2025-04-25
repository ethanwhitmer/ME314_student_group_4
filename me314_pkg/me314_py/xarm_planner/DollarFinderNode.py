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
from groundingdino.util.inference import load_model, predict, load_image
from segment_anything import sam_model_registry, SamPredictor
import torch
import groundingdino.datasets.transforms as T
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

class DollarFinderNode(Node):
    def __init__(self, visualize=False, real=False):
        super().__init__('pick_place_node')

         # Get the package share directory to be used to find model weights
        pkg_path = get_package_share_directory('me314_pkg')
        
        # Process parser arguments
        self.real = real
        self.visualize = visualize

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
        if self.real:
            self.camera_subscription = self.create_subscription(Image,"/camera/realsense2_camera_node/color/image_raw",self.CameraRGBCallback,qos_profile=qos_profile_sensor_data) # RGB Camera
        else:
            self.camera_subscription = self.create_subscription(Image,"/color/image_raw",self.CameraRGBCallback,qos_profile=qos_profile_sensor_data) # RGB Camera
        
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

        # Construct model paths
        GD_model_config = os.path.join(pkg_path, 'models', 'GroundingDINO_SwinT_OGC.py')
        GD_model_weights = os.path.join(pkg_path, 'models', 'groundingdino_swint_ogc.pth')

        # Load Grounding DINO
        self.grounding_dino_model = load_model(
            GD_model_config, GD_model_weights
        )

        # Defining transform for images being input to Grounding Dino
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Load models with device check
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Grounding dino using {device}")

        # Configure model paths
        Sam_model_weights = os.path.join(pkg_path, 'models', 'sam_vit_b_01ec64.pth')

        # Load SAM
        sam = sam_model_registry["vit_b"](checkpoint=Sam_model_weights).to(device)
        print(f"SAM using {next(sam.parameters()).device}") 
        self.sam_predictor = SamPredictor(sam)

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
    
    def AngleBetweenVectors(self,a, b):
        """Compute angle (in degrees) between two vectors a and b."""
        dot = np.dot(a, b)
        det = np.cross(a, b)  # Determinant (a_x*b_y - a_y*b_x)
        angle_rad = np.arctan2(det, dot) 
        
        return np.degrees(angle_rad)
    
    def ConvertBox(self, box, image_shape):
        """
        Convert a SINGLE Grounding DINO box to SAM format
        Args:
            box (torch.Tensor): [4] in [x_center, y_center, width, height] (0..1)
            image_shape (tuple): (height, width)
        Returns:
            np.ndarray: [4] in [x_min, y_min, x_max, y_max] (pixel coordinates)
        """
        h, w = image_shape
        box = box * torch.tensor([w, h, w, h])  # De-normalize
        x_min = box[0] - box[2] / 2
        y_min = box[1] - box[3] / 2
        x_max = box[0] + box[2] / 2
        y_max = box[1] + box[3] / 2
        return np.array([x_min, y_min, x_max, y_max]) 

    def is_fully_visible(self,box, image_width, image_height, margin=0):
        """
        Check if a bounding box is fully inside the frame (with optional margin).
        Args:
            box_xyxy: [x_min, y_min, x_max, y_max] in pixels
            image_width, image_height: Frame dimensions
            margin: Safety margin (pixels) to exclude edge-touching boxes
        Returns:
            bool: True if fully inside frame, False otherwise
        """
        x_min = np.min(box[:, 0])  
        y_min = np.min(box[:, 1])  
        x_max = np.max(box[:, 0])  
        y_max = np.max(box[:, 1])
        return (
            x_min >= margin and
            y_min >= margin and
            x_max <= (image_width - margin) and
            y_max <= (image_height - margin)
        )

    def getObjectPose(self, object):
        dollar_pose = Pose()
        square_point = Point()
        # Case to look for dollar
        if object == "Dollar":
            print("Looking for dollar")
            # Prepare image to input into Grounding Dino
            frame_pil = PIL_Image.fromarray(self.cv_image)
            frame_transformed, _ = self.transform(frame_pil, None)
            
            # Create a copy of the image to be used in sam
            image_sam_input = self.cv_image.copy()

            # Detect objects with Grounding DINO
            boxes, logits, phrases = predict(
                model=self.grounding_dino_model,
                image=frame_transformed,
                caption="dollar bill",
                box_threshold=0.7,
                text_threshold=0.7
            )

            # If dollar is found by grounding dino
            if len(boxes) > 0:
                print("Dollar found")
                # Convert Grounding Dino box format to Sam format
                box_input = self.ConvertBox(boxes[torch.argmax(logits)], image_sam_input.shape[0:2])

                # Segment with SAM
                self.sam_predictor.set_image(image_sam_input)
                masks, _, _ = self.sam_predictor.predict(box=box_input,multimask_output=False)
                mask = masks[0].astype(np.uint8) * 255

                # Find contours from mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
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

                        if not self.is_fully_visible(box, self.cv_image.shape[1], self.cv_image.shape[0], margin=5):
                            print("Dollar not fully in frame")

                            dollar_pose.position.x = float(center_x)
                            dollar_pose.position.y = float(center_y)
                            dollar_pose.position.z = 0.5
                            
                            break
                        else:
                            print("Dollar fully in frame")
                            # Find the bottom left point
                            sorted_by_y = sorted(box, key=lambda p: -p[1])  # Sort by Y descending
                            # Compare the distances of each point to the bottom left point
                            dist = np.linalg.norm(box - sorted_by_y[0], axis=1)
                            # Get the distances of the two closes points
                            closest_points_index =  np.argsort(dist)[1:3]
                            # Create two vectors from the bottom left point of the rectangle to the closest two points
                            vector1 = box[closest_points_index[0]] - sorted_by_y[0]
                            vector2 = box[closest_points_index[1]] - sorted_by_y[0]
                            if np.linalg.norm(vector1) < np.linalg.norm(vector2):
                                vector = vector1
                            else:
                                vector = vector2

                            angle = self.AngleBetweenVectors(vector, np.array([1,0]))
                            cv2.putText(self.cv_image, f"Dollar, angle: {angle:.1f}", tuple(box[0]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                            
                            # Display center coordinates
                            cv2.putText(self.cv_image, f"Center: ({center_x}, {center_y})", 
                                    (center_x + 20, center_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                            dollar_pose.position.x = float(center_x)
                            dollar_pose.position.y = float(center_y)
                            dollar_pose.position.z = 1.0
                            dollar_pose.orientation.z = angle
                            break
            else:
                print("Dollar not found")
                dollar_pose.position.z = 0.0
            self.dollar_pose_publisher.publish(dollar_pose)
        elif object == "Square":
            print("Looking for square")
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2HSV)
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([70, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

             # Clean up the mask
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:  # Skip small detections
                    continue
                
                # Approximate the contour to a polygon
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                
                if len(approx) == 4:
                    print("Square found")
                    # Get the square
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    
                    # Draw the square
                    cv2.drawContours(self.cv_image, [box], 0, (0,255,255), 2)
                    
                    # Calculate center point
                    center_x = int(rect[0][0])
                    center_y = int(rect[0][1])
                    center = (center_x, center_y)
                    
                    # Draw center point
                    cv2.circle(self.cv_image, center, 5, (255, 0, 255), -1)

                    square_point.x = float(center_x)
                    square_point.y = float(center_y)
                    square_point.z = 1.0
                    break
                else:
                    print("Square not found")
                    square_point.z = 0.0            
            self.square_point_publisher.publish(square_point)
        if self.visualize:
            cv2.imshow("Visualization",cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2BGR))
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
    node = DollarFinderNode(
        visualize=known_args.visualize,
        real=known_args.real)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()