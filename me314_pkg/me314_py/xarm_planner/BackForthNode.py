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
FORCE_DETECTED = 3

# Events
ES_TIMEOUT = 11
ES_COMMAND_EXECUTED = 12
ES_COLLISION_DETECTED = 13

class BackForthNode(Node):
    def __init__(self):
        super().__init__('back_forth_node')

        self.position1 = np.array([0.3,0.0,0.1])
        self.position2 = np.array([0.3,0.0,0.3])

        # self.position1 = np.array([0.2,-0.1,0.1])
        # self.position2 = np.array([0.4,0.1,0.1])
        
        # Publishers
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)

        # Subscribers
        self.CommandSubscriber = self.create_subscription(String,"/me314_xarm_current_command",self.GetCurrentCommand,10) # Current Command in Execution
        self.ft_ext_state_sub = self.create_subscription(WrenchStamped, '/xarm/uf_ftsensor_ext_states', self.ForceCallback, 10)
        self.collision_sub = self.create_subscription(Bool, '/me314_xarm_collision', self.CollisionDetected, 10)

        # Initialization of Various Variables
        self.previousCommand = ""
        self.finalCommand = None
        self.baseTransform = None
        self.haveGravityVector = False # Only initialize as True in Sim
        self.gravity = None
        self.currentForce = None
        self.FirstTimeout = False
        self.previousState = None
        self.state = INITIALIZATION

        # Set a timer to repeatedly check if we have Camera Intrinsics and Depth Camera visuals
        self.timer_SM = self.create_timer(2.0, self.TimeoutCallback)
    
    def GetCurrentCommand(self,command):
        #self.get_logger().info(f'Command = {command.data}')
        if command.data != self.previousCommand:
            if self.finalCommand in self.previousCommand and command.data == "":
                self.StateMachine(ES_COMMAND_EXECUTED)
        self.previousCommand = command.data
    
    def CollisionDetected(self,msg):
        if msg.data:
            self.StateMachine(ES_COLLISION_DETECTED)
    
    def TimeoutCallback(self):
        if not self.FirstTimeout:
            self.FirstTimeout = True
        else:
            self.StateMachine(ES_TIMEOUT)
    
    def CancelTimerSM(self):
        self.timer_SM.cancel()
        self.FirstTimeout = False
    
    def ForceCallback(self,msg):
        # Note: this function makes the assumption that the robot maintains the same orientation for the entirety of the file
        if not self.haveGravityVector:
            self.gravity = msg.wrench.force
            self.haveGravityVector = True
        else:
            self.currentForce = msg.wrench.force
            self.currentForce.x -= self.gravity.x
            self.currentForce.y -= self.gravity.y
            self.currentForce.z -= self.gravity.z
    
    def StateMachine(self,Event):
        if self.state == INITIALIZATION:
            if Event == ES_TIMEOUT:
                # The only point of this state is to ensure we have the gravity vector before we start looking for the dollar
                # This state is entered by a timer timing out every 2 seconds
                self.get_logger().info(f'Checking if gravity vector has been obtained')
                if self.haveGravityVector:
                    self.CancelTimerSM()

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

                    self.get_logger().info(f'Gravity vector has been obtained . . . Sending Request to go to Position 1')

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
            elif Event == ES_COLLISION_DETECTED:
                # If a collision is detected, save the current state so we know where to return once the collision is gone
                self.previousState = MOVING_TO_POSITION_1

                # Change state and check every 2 seconds if the collision is gone
                self.timer_SM = self.create_timer(2.0, self.TimeoutCallback)
                self.state = FORCE_DETECTED

                self.get_logger().info(f'Force detected . . . waiting until it is gone before resuming motion')
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
            elif Event == ES_COLLISION_DETECTED:
                # If a collision is detected, save the current state so we know where to return once the collision is gone
                self.previousState = MOVING_TO_POSITION_2

                # Change state and check every 2 seconds if the collision is gone
                self.timer_SM = self.create_timer(2.0, self.TimeoutCallback)
                self.state = FORCE_DETECTED

                self.get_logger().info(f'Force detected . . . waiting until it is gone before resuming motion')
        elif self.state == FORCE_DETECTED:
            if Event == ES_TIMEOUT:
                # If the detected force is small enough, then we can resume motion
                if math.sqrt(self.currentForce.x**2 + self.currentForce.y**2 + self.currentForce.z**2) < 1:
                    # Cancel the timer so we stop receiving timeout events
                    self.CancelTimerSM()

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
                    self.get_logger().info(f'Detected force has left . . . Resuming previous motion')
        

if __name__ == '__main__':
    rclpy.init()
    node = BackForthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()