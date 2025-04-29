#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    # Declare robot_ip argument
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.213', 
        description='IP address of the xArm robot'
    )
    
    # Create LaunchConfiguration objects
    robot_ip = LaunchConfiguration('robot_ip')
    
    # 1) Include the xarm7 MoveIt realmove launch
    xarm_moveit_realmove_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xarm_moveit_config'),
                'launch',
                'xarm7_moveit_realmove.launch.py'
            )
        ),
        launch_arguments={
            'robot_ip': robot_ip,
            'add_gripper': 'true',
            'add_realsense_d435i': 'true',
            'hw_ns': 'xarm',
            'no_gui_ctrl': 'false',
            'report_type': 'rich',
            'add_ft_sensor': 'true',
        }.items()
    )
    
    # 2) Launch xarm commander with a delay
    xarm_pose_commander_node = Node(
        package='me314_pkg',  
        executable='xarm_commander.py',    
        output='screen',
        parameters=[{'use_sim': False}]  # Setting use_sim to False for real hardware
    )

    # 3) Launch the RealSense camera node
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera_node',
        output='screen',
        parameters=[{
            'enable_color': True,
            'enable_depth': True,
            'unite_imu_method': 0,
            'align_depth.enable': True
        }]
    )
    
    # Add a delay before starting the commander
    delayed_commander = TimerAction(period=7.0, actions=[xarm_pose_commander_node])
    
    return LaunchDescription([
        robot_ip_arg,
        xarm_moveit_realmove_launch,
        delayed_commander,
        realsense_node
    ])