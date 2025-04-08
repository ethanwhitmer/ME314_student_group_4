#!/usr/bin/env python3

"""
XArm commander node that implements a command queue system.
Commands are received through a dedicated topic and executed sequentially.
The node only moves to the next command after the current one is fully completed.
"""

import rclpy
import math
import threading
from collections import deque
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String, Bool

# Custom message type for command queue (which contains an array of CommandWrapper messages)
from me314_msgs.msg import CommandQueue

from moveit_msgs.srv import (GetPositionIK, GetCartesianPath, GetMotionPlan, GetPositionFK)
from moveit_msgs.msg import (
    RobotTrajectory,
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    RobotState
)
from moveit_msgs.action import ExecuteTrajectory


class ME314_XArm_Queue_Commander(Node):
    def __init__(self):
        super().__init__('ME314_XArm_Queue_Commander_Node')
        
        # Define ANSI color codes for terminal output
        self.GREEN = "\033[32m"
        self.RED = "\033[31m"
        self.RESET = "\033[0m"
        self.BOLD = "\033[1m"

        ####################################################################
        # QUEUE SYSTEM
        ####################################################################
        self.command_queue = deque()
        self.queue_lock = threading.Lock()
        self.is_executing = False
        self.execution_condition = threading.Condition(self.queue_lock)

        ####################################################################
        # CLIENTS
        ####################################################################
        self.cartesian_path_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.compute_ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_path_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.wait_for_all_services_and_action()

        ####################################################################
        # CLASS ATTRIBUTES
        ####################################################################
        self.declare_parameter('use_sim', False)
        self.use_sim = self.get_parameter('use_sim').value
        self.log_info(f"Running with use_sim={self.use_sim}")

        self.current_gripper_position = 0.0
        self.home_joints_deg = [1.1, -48.5, 0.4, 32.8, 0.4, 81.9, 0.3]
        self.home_joints_rad = [math.radians(angle) for angle in self.home_joints_deg]
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.current_joint_positions = [None] * len(self.joint_names)

        self.gripper_group_name = "xarm_gripper"  
        self.gripper_joint_names = ["drive_joint"]
        
        self.initialization_complete = False

        # Define the workspace bounds in millimeters
        self.x_bounds = (150, 500)   
        self.y_bounds = (-300, 300) 
        if self.use_sim:
            self.z_bounds = (10, 400) 
        else:
            self.z_bounds = (35, 400)    

        # Tracking how many commands get rejected by manual bound checks
        self.rejected_commands_count = 0

        ####################################################################
        # SUBSCRIBERS
        ####################################################################
        self.queue_cmd_sub = self.create_subscription(CommandQueue, '/me314_xarm_command_queue', self.command_queue_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        ####################################################################
        # PUBLISHERS
        ####################################################################
        self.current_pose_pub = self.create_publisher(Pose, '/me314_xarm_current_pose', 10)
        self.curr_joint_position_deg_pub = self.create_publisher(JointState, '/me314_xarm_current_joint_positions_deg', 10)
        self.gripper_position_pub = self.create_publisher(Float64, '/me314_xarm_gripper_position', 10)

        self.queue_size_pub = self.create_publisher(Float64, '/me314_xarm_queue_size', 10)
        self.is_executing_pub = self.create_publisher(Bool, '/me314_xarm_is_executing', 10)
        self.current_command_pub = self.create_publisher(String, '/me314_xarm_current_command', 10)
        self.rejected_command_pub = self.create_publisher(String, '/me314_xarm_rejected_command', 10)

        # Timers for status publishing and processing the queue
        self.timer_period = 0.1  # seconds
        self.timer_pose = self.create_timer(self.timer_period, self.publish_current_pose)
        self.timer_gripper = self.create_timer(self.timer_period, self.publish_gripper_position)
        self.timer_joint_positions = self.create_timer(self.timer_period, self.publish_current_joint_positions)
        self.timer_queue_status = self.create_timer(self.timer_period, self.publish_queue_status)
        self.timer_queue_processor = self.create_timer(self.timer_period, self.process_command_queue)

        ####################################################################
        # INITIALIZATION
        ####################################################################
        self.log_info("Moving to home position (joint-based).")
        self.plan_execute_joint_target_async(self.home_joints_rad, callback=self.home_move_done_callback)
        self.log_info("Opening gripper fully.")
        self.plan_execute_gripper_async(0.0, callback=self.init_done_callback)
        self.log_info("XArm queue commander node is starting initialization...")

        # Log the workspace bounds
        self.log_info(f"Workspace bounds initialized: X={self.x_bounds}mm, Y={self.y_bounds}mm, Z={self.z_bounds}mm")

    ####################################################################
    # LOGGING METHODS
    ####################################################################
    def log_info(self, message):
        """Log information with green color"""
        colored_message = f"{self.GREEN}[XArm] {message}{self.RESET}"
        self.get_logger().info(colored_message)
    
    def log_warn(self, message):
        """Log warnings with red color"""
        colored_message = f"{self.RED}{self.BOLD}[XArm] WARNING: {message}{self.RESET}"
        self.get_logger().warn(colored_message)
    
    def log_error(self, message):
        """Log errors with red color and bold"""
        colored_message = f"{self.RED}{self.BOLD}[XArm] ERROR: {message}{self.RESET}"
        self.get_logger().error(colored_message)

    ####################################################################
    # BOUNDS CHECKING
    ####################################################################
    def is_pose_within_bounds(self, pose: Pose) -> bool:
        """
        Check if the given pose is within the manual bounding region in millimeters.
        Return True if valid, False otherwise (and log a warning).
        """
        x_mm = pose.position.x * 1000
        y_mm = pose.position.y * 1000
        z_mm = pose.position.z * 1000

        within_x = self.x_bounds[0] <= x_mm <= self.x_bounds[1]
        within_y = self.y_bounds[0] <= y_mm <= self.y_bounds[1]
        within_z = self.z_bounds[0] <= z_mm <= self.z_bounds[1]

        if not (within_x and within_y and within_z):
            reasons = []
            if not within_x:
                reasons.append(f"X ({x_mm:.1f}mm) outside [{self.x_bounds[0]}, {self.x_bounds[1]}]mm")
            if not within_y:
                reasons.append(f"Y ({y_mm:.1f}mm) outside [{self.y_bounds[0]}, {self.y_bounds[1]}]mm")
            if not within_z:
                reasons.append(f"Z ({z_mm:.1f}mm) outside [{self.z_bounds[0]}, {self.z_bounds[1]}]mm")

            self.log_warn("POSE OUT OF BOUNDS: " + ", ".join(reasons))
            rejected_msg = String()
            rejected_msg.data = (
                f"REJECTED: Pose [X={x_mm:.1f}, Y={y_mm:.1f}, Z={z_mm:.1f}]mm - " +
                ", ".join(reasons)
            )
            self.rejected_command_pub.publish(rejected_msg)
            self.rejected_commands_count += 1
            return False
        return True

    ####################################################################
    # QUEUE METHODS
    ####################################################################
    def init_done_callback(self, success: Bool):
        if success:
            self.log_info("Initialization complete. Ready to process commands.")
            self.initialization_complete = True

        else:
            self.log_warn("Initialization failed. Gripper open command failed.")
            self.initialization_complete = True

    def command_queue_callback(self, msg: CommandQueue):
        with self.queue_lock:
            for command in msg.commands:
                if command.command_type == "pose":
                    pose_cmd = command.pose_command
                    pose = Pose()
                    pose.position.x = pose_cmd.x
                    pose.position.y = pose_cmd.y
                    pose.position.z = pose_cmd.z
                    pose.orientation.x = pose_cmd.qx
                    pose.orientation.y = pose_cmd.qy
                    pose.orientation.z = pose_cmd.qz
                    pose.orientation.w = pose_cmd.qw

                    # Manual bounding check
                    if self.is_pose_within_bounds(pose):
                        self.command_queue.append(("pose", pose))
                        self.log_info(
                            f"Queued pose command: "
                            f"[{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}]"
                        )
                    else:
                        self.log_warn(
                            f"Rejected pose command: "
                            f"[{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}]"
                            " - Outside workspace bounds"
                        )

                elif command.command_type == "gripper":
                    gripper_cmd = command.gripper_command
                    self.command_queue.append(("gripper", gripper_cmd.gripper_position))
                    self.log_info(f"Queued gripper command: {gripper_cmd.gripper_position}")

                elif command.command_type == "joint":
                    joint_cmd = command.joint_command
                    joint_positions = [
                        joint_cmd.joint1, joint_cmd.joint2, joint_cmd.joint3,
                        joint_cmd.joint4, joint_cmd.joint5, joint_cmd.joint6, joint_cmd.joint7
                    ]
                    self.command_queue.append(("joint", joint_positions))
                    self.log_info(
                        f"Queued joint command: {[math.degrees(j) for j in joint_positions]}"
                    )

                elif command.command_type == "home":
                    self.command_queue.append(("home", None))
                    self.log_info("Queued home command")
                else:
                    self.log_warn(f"Unknown command type: {command.command_type}")

            self.execution_condition.notify()

        self.log_info(
            f"Added {len(msg.commands)} commands to queue. Current queue size: {len(self.command_queue)}"
        )

    def process_command_queue(self):
        if not self.initialization_complete:
            return

        with self.queue_lock:
            if not self.command_queue or self.is_executing:
                return

            command_type, command_data = self.command_queue[0]
            self.is_executing = True

        if command_type == "pose":
            self.log_info(
                f"Executing pose command: "
                f"[{command_data.position.x:.3f}, {command_data.position.y:.3f}, {command_data.position.z:.3f}]"
            )
            self.compute_ik_and_execute_joint_async(command_data, callback=self.command_execution_complete)

        elif command_type == "gripper":
            self.log_info(f"Executing gripper command: {command_data}")
            self.plan_execute_gripper_async(command_data, callback=self.command_execution_complete)

        elif command_type == "joint":
            self.log_info(
                f"Executing joint command: {[math.degrees(j) for j in command_data]}"
            )
            self.plan_execute_joint_target_async(command_data, callback=self.command_execution_complete)

        elif command_type == "home":
            self.log_info("Executing home command")
            self.plan_execute_joint_target_async(self.home_joints_rad, callback=self.command_execution_complete)

    def command_execution_complete(self, success: bool):
        with self.queue_lock:
            if success:
                self.log_info("Command executed successfully")
                if self.command_queue:
                    self.command_queue.popleft()
            else:
                self.log_warn("Command execution failed")
            self.is_executing = False

    def publish_queue_status(self):
        with self.queue_lock:
            queue_size = len(self.command_queue)
            is_executing = self.is_executing
            current_command = ""
            if queue_size > 0:
                cmd_type, cmd_data = self.command_queue[0]
                if cmd_type == "pose":
                    current_command = (
                        f"Pose: [{cmd_data.position.x:.3f}, {cmd_data.position.y:.3f}, {cmd_data.position.z:.3f}]"
                    )
                elif cmd_type == "gripper":
                    current_command = f"Gripper: {cmd_data}"
                elif cmd_type == "joint":
                    current_command = f"Joint: {[math.degrees(j) for j in cmd_data]}"
                elif cmd_type == "home":
                    current_command = "Home"

        queue_size_msg = Float64()
        queue_size_msg.data = float(queue_size)
        self.queue_size_pub.publish(queue_size_msg)

        is_executing_msg = Bool()
        is_executing_msg.data = is_executing
        self.is_executing_pub.publish(is_executing_msg)

        current_command_msg = String()
        current_command_msg.data = current_command
        self.current_command_pub.publish(current_command_msg)

    ####################################################################
    # CORE METHODS
    ####################################################################
    def wait_for_all_services_and_action(self):
        while not self.cartesian_path_client.wait_for_service(timeout_sec=1.0):
            self.log_info('Waiting for compute_cartesian_path service...')
        while not self.compute_ik_client.wait_for_service(timeout_sec=1.0):
            self.log_info('Waiting for compute_ik service...')
        while not self.plan_path_client.wait_for_service(timeout_sec=1.0):
            self.log_info('Waiting for plan_kinematic_path service...')
        while not self.execute_client.wait_for_server(timeout_sec=1.0):
            self.log_info('Waiting for execute_trajectory action server...')
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.log_info('Waiting for compute_fk service...')

        self.log_info('All services and action servers are available!')

    def home_move_done_callback(self, success: bool):
        if not success:
            self.log_warn("Failed to move to home position (joint-based).")
        else:
            self.log_info("Home position move completed successfully (joint-based).")

    def joint_state_callback(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            if name in self.joint_names:
                i = self.joint_names.index(name)
                self.current_joint_positions[i] = position
            if name == "xarm_gripper_drive_joint":
                self.current_gripper_position = position

    def publish_current_joint_positions(self):
        if None in self.current_joint_positions:
            return
        msg = JointState()
        msg.name = self.joint_names
        msg.position = [math.degrees(pos) for pos in self.current_joint_positions]
        self.curr_joint_position_deg_pub.publish(msg)

    def publish_current_pose(self):
        if None in self.current_joint_positions:
            return
        robot_state = RobotState()
        robot_state.joint_state.name = self.joint_names
        robot_state.joint_state.position = self.current_joint_positions

        req = GetPositionFK.Request()
        req.header.frame_id = "link_base"
        req.fk_link_names = ["link_tcp"]
        req.robot_state = robot_state

        future = self.fk_client.call_async(req)
        future.add_done_callback(self.publish_current_pose_cb)

    def publish_current_pose_cb(self, future):
        try:
            res = future.result()
            if res is not None and len(res.pose_stamped) > 0:
                ee_pose = res.pose_stamped[0].pose
                self.current_pose_pub.publish(ee_pose)
        except Exception as e:
            self.log_error(f"FK service call failed: {e}")

    def publish_gripper_position(self):
        if self.current_gripper_position is None:
            return
        msg = Float64()
        msg.data = self.current_gripper_position
        self.gripper_position_pub.publish(msg)

    def compute_ik_and_execute_joint_async(self, target_pose: Pose, callback=None):
        # For real hardware, apply an additional +0.058m offset to Z
        if not self.use_sim:
            self.log_info("Applying +0.058 m offset in Z because use_sim=False")
            target_pose.position.z += 0.058

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = "xarm7"
        ik_req.ik_request.robot_state.is_diff = True
        ik_req.ik_request.pose_stamped.header.frame_id = "link_base"
        ik_req.ik_request.pose_stamped.pose = target_pose
        ik_req.ik_request.timeout.sec = 2

        ik_req.ik_request.avoid_collisions = False

        # No collision checking with bounding boxes; we do manual checks
        future_ik = self.compute_ik_client.call_async(ik_req)
        future_ik.add_done_callback(lambda f: self.compute_ik_done_cb(f, callback))

    def compute_ik_done_cb(self, future, callback=None):
        try:
            res = future.result()
        except Exception as e:
            self.log_error(f"IK call failed: {e}")
            if callback:
                callback(False)
            return

        if res.error_code.val != 1:
            self.log_warn(f"IK did not succeed, error code: {res.error_code.val}")
            if callback:
                callback(False)
            return

        joint_solution = res.solution.joint_state
        desired_positions = [0.0] * len(self.joint_names)
        for name, pos in zip(joint_solution.name, joint_solution.position):
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                desired_positions[idx] = pos

        self.log_info("IK succeeded; now planning joint motion to that IK solution.")
        self.plan_execute_joint_target_async(desired_positions, callback=callback)

    def plan_execute_joint_target_async(self, joint_positions, callback=None):
        req = GetMotionPlan.Request()
        motion_req = MotionPlanRequest()
        motion_req.workspace_parameters.header.frame_id = "link_base"
        motion_req.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
        motion_req.start_state.is_diff = True
        motion_req.goal_constraints.append(Constraints())

        for i, joint_name in enumerate(self.joint_names):
            constraint = JointConstraint()
            constraint.joint_name = joint_name
            constraint.position = joint_positions[i]
            constraint.tolerance_above = 0.01
            constraint.tolerance_below = 0.01
            constraint.weight = 1.0
            motion_req.goal_constraints[0].joint_constraints.append(constraint)

        motion_req.group_name = "xarm7"
        motion_req.num_planning_attempts = 10
        motion_req.allowed_planning_time = 5.0
        motion_req.path_constraints.name = "disable_collisions"

        if self.use_sim:
            motion_req.max_velocity_scaling_factor = 0.2
            motion_req.max_acceleration_scaling_factor = 0.2
        else:
            motion_req.max_velocity_scaling_factor = 0.08
            motion_req.max_acceleration_scaling_factor = 0.08

        req.motion_plan_request = motion_req

        pos_deg = [math.degrees(a) for a in joint_positions]
        self.log_info(f'Planning joint motion to positions (deg): {pos_deg}')

        future = self.plan_path_client.call_async(req)
        future.add_done_callback(lambda f: self.plan_path_done_cb(f, callback))

    def plan_path_done_cb(self, future, callback):
        try:
            result = future.result()
        except Exception as e:
            self.log_error(f"Joint path plan service call failed: {e}")
            if callback:
                callback(False)
            return

        if result.motion_plan_response.error_code.val != 1:
            self.log_error(
                f"Planning failed, error code = {result.motion_plan_response.error_code.val}"
            )
            if callback:
                callback(False)
            return

        self.log_info("Joint motion plan succeeded, executing trajectory...")
        self.execute_trajectory_async(result.motion_plan_response.trajectory, callback)

    def execute_trajectory_async(self, trajectory: RobotTrajectory, callback=None):
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = trajectory

        self.log_info("Sending trajectory for execution...")
        send_goal_future = self.execute_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(lambda f: self.action_server_send_callback(f, callback))

    def action_server_send_callback(self, future, callback):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.log_error("ExecuteTrajectory goal was rejected by server.")
            if callback:
                callback(False)
            return

        self.log_info("Goal accepted by server, waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self.action_server_execute_callback(f, callback))

    def action_server_execute_callback(self, future, callback):
        result = future.result().result
        if result.error_code.val != 1:
            self.log_error(f"Trajectory execution failed with error code: {result.error_code.val}")
            if callback:
                callback(False)
            return

        self.log_info("Trajectory execution succeeded.")
        if callback:
            callback(True)

    def plan_execute_gripper_async(self, position: float, callback=None):
        req = GetMotionPlan.Request()
        motion_req = MotionPlanRequest()
        motion_req.workspace_parameters.header.frame_id = "link_base"
        motion_req.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
        motion_req.start_state.is_diff = True
        motion_req.goal_constraints.append(Constraints())

        for jn in self.gripper_joint_names:
            constraint = JointConstraint()
            constraint.joint_name = jn
            constraint.position = position
            constraint.tolerance_above = 0.01
            constraint.tolerance_below = 0.01
            constraint.weight = 1.0
            motion_req.goal_constraints[0].joint_constraints.append(constraint)

        motion_req.group_name = self.gripper_group_name
        motion_req.num_planning_attempts = 10
        motion_req.allowed_planning_time = 5.0
        motion_req.max_velocity_scaling_factor = 0.1
        motion_req.max_acceleration_scaling_factor = 0.1

        req.motion_plan_request = motion_req

        self.log_info(f"Planning gripper motion to {math.degrees(position):.2f} deg")

        future = self.plan_path_client.call_async(req)
        future.add_done_callback(lambda f: self.plan_path_done_cb(f, callback))


def main(args=None):
    rclpy.init(args=args)
    commander = ME314_XArm_Queue_Commander()

    try:
        rclpy.spin(commander)
    except KeyboardInterrupt:
        pass

    commander.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()