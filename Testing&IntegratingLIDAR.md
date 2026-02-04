# Testing LIDAR individually- Objective
To test lidar individually first we need to make sure lidar runs correctly, so we test with PI 5 separately  
Once done making sure lidar operates correctly, then we need to use a virtual environment first, then apply the ROS environment, ROS environment is Jeffry's task, coordinate with Cristian for help 
Link is:  
https://www.youtube.com/watch?v=OSoMSVry-8E  

Using the Hypersen github for this specific lidar sensor, we are not able to run the demo nor the software on the RPI5 due to the incompatibility being the "libhps3d.so" file used to run the program is formatted for x86-64 Intel/AMD while our RPI5 is aarch64(ARM64) thus we need to find a SDK that is compatible with the ARM64. Same can be said for Jetson Orin Nano for the architecture is ARM64.

## First task- Get offical SDK clone and test

## Second task- Intehgrate ROS2 for rover to use sensor once lidar is already tested with code functioning as intended

