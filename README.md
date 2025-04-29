# ME314 XArm Control Package
### Overview
The ME314_XArm package provides Python-based control and teleoperation functionalities for the xArm7 robotic arm, integrating with ROS2 for Stanford University's ME 314: Robotic Dexterity taught by Dr. Monroe Kennedy III.

### Caution
Please stay away from the robot arm during operation to avoid personal injury or equipment damage.
Ensure e-stop is close and accessible before controlling arm.

### Setting up Docker Image (for Windows and Mac users)
1. Install Docker Desktop (or Docker Engine for CLI only)
2. In terminal run: 

```bash
docker pull aqiu218/me314_xarm_ros2
```

4. Clone ME314_student repo to your computer

```bash
git clone https://github.com/armlabstanford/ME314_student.git
```

3. Start container using the following command (only needs to be run once): 

docker run --privileged --name me314_ros2 -p 6080:80 --shm-size=512m -v _**<path/to/ME314_student>**:_/home/ubuntu/xarm_ros2_ws/src/me314 aqiu218/me314_xarm_ros2

** In the above command, *-v <computer-path>:<docker-path>* mounts a folder on your local computer to the docker container, thus linking any files/directories/changes made on local to your docker container ubuntu system. 
** --name sets the name of the container, this can be set to anything you want. We want to link our ME314_student folder from our host device to the virtualized ubuntu system.

Example:

```bash
docker run --privileged --name me314_ros2 -p 6080:80 --shm-size=512m -v /home/alex/ME314_student:/home/ubuntu/xarm_ros2_ws/src/me314 aqiu218/me314_xarm_ros2
```

** WE HIGHLY RECOMMEND ONLY CHANGING THE COMPUTER PATH TO YOUR ME314_student REPO IN THE ABOVE EXAMPLE COMMAND **



4. Navigate to http://localhost:6080/ and click connect. You should now see a full Ubuntu Linux desktop environment!

5. Stop container by pressing ctrl+c in host device (your laptop) terminal

6. To run container in the future, run the following command in your terminal and navigate to http://localhost:6080/:

```bash
docker start me314_ros2
```
7. To stop container (run in terminal): 

```bash
docker stop me314_ros2
```

### Testing out the me314_pkg!

1. Navigate to terminal (if using native Linux) or double click on Terminator on home screen.

2. Build your ROS2 workspace (you should only need to build the first time you start your docker container):

```bash
cd /home/ubuntu/xarm_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select me314_pkg me314_msgs
source install/setup.bash
```

You will see some warnings, these can be ignored.

3. To start gazebo simulation (and RViz), run the following launch command in the same terminal:

```bash
ros2 launch me314_pkg me314_xarm_gazebo.launch.py
```

This will take a while and will likely fail the first time you run this command when starting the container since Gazebo loads very slowly, ctrl+c and re-run the command after gazebo fully opens but no robot appears. Now you should see the xarm7 spawn on a table, with a red block in front of it.

4. To test an example script that commands the xarm to move from point A to B, run the following command in a separate terminal while gazebo and rviz are open:

```bash
cd xarm_ros2_ws
source install/setup.bash
clear
ros2 run me314_pkg xarm_a2b_example.py
```

### Installation (if using native Linux Ubuntu 22.04 System)

#### Install ros2 humble (for Ubuntu 22.04)
Follow instructions for ros2 humble (desktop) install: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html or copy and paste the below commands in terminal:

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop
```

#### Install Gazebo/ROS2 Controllers

```bash
sudo apt install gazebo
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install \
  ros-humble-gazebo-ros2-control \
  ros-humble-controller-manager \
  ros-humble-ros2-control \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller
```

#### Install Realsense2 SDK and ROS2 Wrapper

a. Install librealsense (source: https://github.com/IntelRealSense/realsense-ros?tab=readme-ov-file#installation-on-ubuntu step #2 option #2)

```bash
# Configure Ubuntu Repositories
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
# Install librealsense2 debian package
sudo apt install ros-humble-librealsense2*
```

b. Install RealSense Wrapper (source: https://github.com/IntelRealSense/realsense-ros?tab=readme-ov-file#installation-on-ubuntu)

```bash
# Assuming Ubuntu Repositories are already configured from previous step, install realsense2 wrapper debian package
sudo apt install ros-humble-realsense2-*
```

#### Install Moveit2

```bash
sudo apt install ros-humble-moveit
```

#### Create xarm_ros2_ws

```bash
cd ~
mkdir -p xarm_ros2_ws/src
```

#### Clone xarm_ros2 and build

```bash
cd ~/xarm_ros2_ws/src
git clone https://github.com/Realexqiu/xarm_ros2.git --recursive -b $ROS_DISTRO
rosdep update && rosdep install --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y --skip-keys="roscpp catkin"
cd ~/xarm_ros2_ws
colcon build
```

#### Clone ME314_student and build

```bash
cd ~/xarm_ros2_ws/src
git clone https://github.com/armlabstanford/ME314_student.git
cd ~/xarm_ros2_ws
colcon build --symlink-install --packages-select me314_pkg me314_msgs
```

### Tips/Notes

- If using Terminator, ctrl+shift+E is shortkey to open side by side terminal tab
- If xarm isn't spawning in gazebo, try quitting and re-running launch command
- If encountering the following issue when running a script in docker: **/usr/bin/env: 'python3\r': No such file or directory**, open the file in vscodium and convert the file format from CRLF to LF (bottom right of vscodium)
- For more info about docker check out this quickstart guide: https://github.com/armlabstanford/collaborative-robotics/wiki/Docker-Quickstart
- Docker cheat sheet commands here: https://docs.docker.com/get-started/docker_cheatsheet.pdf

### Commands Summary
#### Navigate to Workspace and Source Install Before Running Any Commands

```bash
cd xarm_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

#### Tele-Operation with Spacemouse (this only works when connected to real-robot and spacemouse (https://3dconnexion.com/us/product/spacemouse-compact/)

```bash
ros2 run me314_pkg xarm_spacemouse_ros2.py
```

#### Control XArm using XArm Planner (with MoveIt API)

1. Control in Gazebo

a. In one terminal run the following command:

```bash
ros2 launch me314_pkg me314_xarm_gazebo.launch.py
```

b. In another terminal run script (example):

```bash
ros2 run me314_pkg xarm_a2b_example.py
```

2. Control in Real

a. In one terminal run the xarm planner launch command:

```bash
ros2 launch me314_pkg me314_xarm_real.launch.py
```

b. In another terminal run script (example):

```bash
ros2 run me314_pkg xarm_a2b_example.py
```

