#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped

class FTMonitor(Node):
    """
    Force/Torque Monitor Node.
    
    This node subscribes to force/torque sensor data and logs the values at a fixed rate.
    It demonstrates:
    - How to subscribe to sensor topics in ROS2
    - How to use timers for periodic operations
    - How to handle and process sensor data
    """
    def __init__(self):
        # Initialize the ROS2 node with a specific name
        super().__init__('FT_Monitor_Node')
        
        # Initialize variables to store the latest force/torque data
        self.FT_force_x = 0.0
        self.FT_force_y = 0.0
        self.FT_force_z = 0.0
        self.FT_torque_x = 0.0
        self.FT_torque_y = 0.0
        self.FT_torque_z = 0.0
        
        # Create a subscription to the force/torque sensor topic
        self.ft_ext_state_sub = self.create_subscription(WrenchStamped, '/xarm/uf_ftsensor_ext_states', self.ft_ext_state_cb, 10)
        
        # Create a timer that calls log_ft_data every 1.0 seconds (1 Hz)
        self.timer = self.create_timer(1.0, self.log_ft_data)
        
        # Log a message to indicate the node has started
        self.get_logger().info("FT Monitor started - logging at 1 Hz")

    def ft_ext_state_cb(self, msg: WrenchStamped):
        """
        Callback function that runs whenever a new force/torque message is received.
        
        This function extracts the force and torque data from the message
        and stores it for later use.
        
        Args:
            msg (WrenchStamped): The force/torque sensor message
        """
        # Extract force components from the message
        self.FT_force_x = msg.wrench.force.x
        self.FT_force_y = msg.wrench.force.y
        self.FT_force_z = msg.wrench.force.z
        
        # Extract torque components from the message
        self.FT_torque_x = msg.wrench.torque.x
        self.FT_torque_y = msg.wrench.torque.y
        self.FT_torque_z = msg.wrench.torque.z

    def log_ft_data(self):
        """
        Timer callback function that logs force/torque data at a fixed rate (1 Hz).
        
        This function:
        1. Calculates the magnitude of force and torque vectors
        2. Logs the individual components and magnitudes
        """
        # Calculate force magnitude using the Euclidean norm (square root of sum of squares)
        force_magnitude = math.sqrt(self.FT_force_x**2 + self.FT_force_y**2 + self.FT_force_z**2)
        
        # Calculate torque magnitude
        torque_magnitude = math.sqrt(self.FT_torque_x**2 + self.FT_torque_y**2 + self.FT_torque_z**2)
        
        # Log the force data (components and magnitude)
        self.get_logger().info(f"Force: [{self.FT_force_x:.2f}, {self.FT_force_y:.2f}, {self.FT_force_z:.2f}] N")
        self.get_logger().info(f"Force magnitude: {force_magnitude:.2f} N")
        
        # Log the torque data (components and magnitude)
        self.get_logger().info(f"Torque: [{self.FT_torque_x:.2f}, {self.FT_torque_y:.2f}, {self.FT_torque_z:.2f}] Nm")
        self.get_logger().info(f"Torque magnitude: {torque_magnitude:.2f} Nm")


def main(args=None):   

    rclpy.init(args=args)
    node = FTMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()