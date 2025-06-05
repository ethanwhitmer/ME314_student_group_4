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

import time
import math

# Frame Imports
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration

class TransformTimerNode(Node):
    def __init__(self):
        super().__init__('transform_timer_node')

        # Force Subscriber
        self.ft_ext_state_sub = self.create_subscription(WrenchStamped, '/xarm/uf_ftsensor_ext_states', self.ForceCallback, 10)

        # Frame Listener Initialization
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialization of Variables
        self.currentForce = None
        self.currentTorque = None
        self.haveForce = False

        self.timer_SM = self.create_timer(1.0, self.TimeoutCallback)
    
    def TimeoutCallback(self):
        if self.haveForce:
            startTime = self.get_clock().now().nanoseconds
            target_frame = 'link_base'
            source_frame = 'ft_sensor_link'
            self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
            # Don't move onto the next part in the code until we've found the transformation matrix
            while not haveTransform:
                self.baseTransform, haveTransform = self.GetTransform(target_frame, source_frame)
            baseForce = self.baseTransform[:,:-1] @ self.currentForce
            baseTorque = self.baseTransform[:,:-1] @ self.currentTorque
            forceNorm = np.linalg.norm(baseForce)
            stopTime = self.get_clock().now().nanoseconds
            deltaTime = (stopTime - startTime) / 1e9
            self.get_logger().info(f'Execution Time: {deltaTime:.2f} seconds')
            self.get_logger().info(f'Force Magnitude: {forceNorm:.2f} N')
            self.get_logger().info(f'Current Base Force: x={baseForce[0]:.2f}, y={baseForce[1]:.2f}, z={baseForce[2]:.2f}')
            self.get_logger().info(f'Current Base Torque: x={baseTorque[0]:.2f}, y={baseTorque[1]:.2f}, z={baseTorque[2]:.2f}')
            self.get_logger().info(f'')
    
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
    
    def ForceCallback(self,msg):
        self.currentForce = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        self.currentTorque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        self.haveForce = True


if __name__ == '__main__':
    rclpy.init()
    node = TransformTimerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()