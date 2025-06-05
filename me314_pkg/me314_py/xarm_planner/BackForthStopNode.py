#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import WrenchStamped

from std_msgs.msg import Float64, String, Bool

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from cv_bridge import CvBridge
import time
import math

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper
from sensor_msgs.msg import JointState

# States
INITIALIZATION = 0
MOVING_TO_POSITION_1 = 1
MOVING_TO_POSITION_2 = 2
STOPPED = 3

# Events
ES_TIMEOUT = 11
ES_COMMAND_EXECUTED = 12

class BackForthStopNode(Node):
    def __init__(self):
        super().__init__('back_forth_stop_node')

        self.position1 = np.array([0.3,0.0,0.1])
        self.position2 = np.array([0.3,0.0,0.3])

        # self.position1 = np.array([0.2,-0.1,0.1])
        # self.position2 = np.array([0.4,0.1,0.1])
        
        # Publishers
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)
        self.cancel_command_pub = self.create_publisher(Bool, '/me314_xarm_cancel_command', 10)

        # Subscribers
        self.CommandSubscriber = self.create_subscription(String,"/me314_xarm_current_command",self.GetCurrentCommand,10) # Current Command in Execution
        self.collision_sub = self.create_subscription(Bool, '/me314_xarm_collision', self.CollisionDetected, 10)

        # Initialization of Various Variables
        self.previousCommand = ""
        self.finalCommand = None
        self.FirstTimeout = True # This makes instant start
        self.previousState = None
        self.state = INITIALIZATION

        # Set a timer to repeatedly check if we have Camera Intrinsics and Depth Camera visuals
        self.timer_SM = self.create_timer(5.0, self.TimeoutCallback)
    
    def GetCurrentCommand(self,command):
        #self.get_logger().info(f'Command = {command.data}')
        if command.data != self.previousCommand:
            if self.finalCommand in self.previousCommand and command.data == "":
                self.StateMachine(ES_COMMAND_EXECUTED)
        self.previousCommand = command.data
    
    def CollisionDetected(self,msg):
        if msg.data:
            self.get_logger().info(f'Successfully canceled a motion')
    
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
                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                # Create a CommandWrapper for the pose command
                wrapper_up = CommandWrapper()
                wrapper_up.command_type = "pose"
                wrapper_up.pose_command.x = self.position1[0]
                wrapper_up.pose_command.y = self.position1[1]
                wrapper_up.pose_command.z = self.position1[2]
                wrapper_up.pose_command.qx = 1.0
                wrapper_up.pose_command.qy = 0.0
                wrapper_up.pose_command.qz = 0.0
                wrapper_up.pose_command.qw = 0.0

                # Specify the final command in this queue for re-entry purposes
                self.finalCommand = "Pose"

                queue_msg.commands.append(wrapper_up)
                self.command_queue_pub.publish(queue_msg)

                self.get_logger().info(f'Sending Request to go to Position 1')

                self.state = MOVING_TO_POSITION_1
        elif self.state == MOVING_TO_POSITION_1:
            if Event == ES_COMMAND_EXECUTED:
                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                # Create a CommandWrapper for the pose command
                wrapper_up = CommandWrapper()
                wrapper_up.command_type = "pose"
                wrapper_up.pose_command.x = self.position2[0]
                wrapper_up.pose_command.y = self.position2[1]
                wrapper_up.pose_command.z = self.position2[2]
                wrapper_up.pose_command.qx = 1.0
                wrapper_up.pose_command.qy = 0.0
                wrapper_up.pose_command.qz = 0.0
                wrapper_up.pose_command.qw = 0.0

                # Specify the final command in this queue for re-entry purposes
                self.finalCommand = "Pose"

                queue_msg.commands.append(wrapper_up)
                self.command_queue_pub.publish(queue_msg)

                self.get_logger().info(f'Position 1 Reached . . . Sending Request to go to Position 2')

                self.state = MOVING_TO_POSITION_2
            elif Event == ES_TIMEOUT:
                msg = Bool()
                msg.data = True
                self.cancel_command_pub.publish(msg)
                self.previousState = MOVING_TO_POSITION_1
                self.state = STOPPED
                self.get_logger().info(f'Stopping motion')
        elif self.state == MOVING_TO_POSITION_2:
            if Event == ES_COMMAND_EXECUTED:
                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                # Create a CommandWrapper for the pose command
                wrapper_up = CommandWrapper()
                wrapper_up.command_type = "pose"
                wrapper_up.pose_command.x = self.position1[0]
                wrapper_up.pose_command.y = self.position1[1]
                wrapper_up.pose_command.z = self.position1[2]
                wrapper_up.pose_command.qx = 1.0
                wrapper_up.pose_command.qy = 0.0
                wrapper_up.pose_command.qz = 0.0
                wrapper_up.pose_command.qw = 0.0

                # Specify the final command in this queue for re-entry purposes
                self.finalCommand = "Pose"

                queue_msg.commands.append(wrapper_up)
                self.command_queue_pub.publish(queue_msg)

                self.get_logger().info(f'Position 2 Reached . . . Sending Request to go to Position 1')

                self.state = MOVING_TO_POSITION_1
            elif Event == ES_TIMEOUT:
                msg = Bool()
                msg.data = True
                self.cancel_command_pub.publish(msg)
                self.previousState = MOVING_TO_POSITION_2
                self.state = STOPPED
                self.get_logger().info(f'Stopping motion')
        elif self.state == STOPPED:
            if Event == ES_TIMEOUT:
                # Create a CommandQueue message
                queue_msg = CommandQueue()
                queue_msg.header.stamp = self.get_clock().now().to_msg()

                # Create a CommandWrapper for the pose command
                wrapper_up = CommandWrapper()
                wrapper_up.command_type = "pose"
                wrapper_up.pose_command.qx = 1.0
                wrapper_up.pose_command.qy = 0.0
                wrapper_up.pose_command.qz = 0.0
                wrapper_up.pose_command.qw = 0.0
                
                # Resume the appropriate motion
                if self.previousState == MOVING_TO_POSITION_1:
                    wrapper_up.pose_command.x = self.position1[0]
                    wrapper_up.pose_command.y = self.position1[1]
                    wrapper_up.pose_command.z = self.position1[2]
                elif self.previousState == MOVING_TO_POSITION_2:
                    wrapper_up.pose_command.x = self.position2[0]
                    wrapper_up.pose_command.y = self.position2[1]
                    wrapper_up.pose_command.z = self.position2[2]
                else:
                    self.get_logger().warn(f'Invalid previous state . . . check code')
                
                # Specify the final command in this queue for re-entry purposes
                self.finalCommand = "Pose"

                queue_msg.commands.append(wrapper_up)
                self.command_queue_pub.publish(queue_msg)

                self.state = self.previousState
                self.get_logger().info(f'Resuming previous motion')
        

if __name__ == '__main__':
    rclpy.init()
    node = BackForthStopNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()