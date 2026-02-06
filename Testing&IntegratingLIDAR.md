# Testing LIDAR individually- Objective
To test lidar individually first we need to make sure lidar runs correctly, so we test with PI 5 separately  
Once done making sure lidar operates correctly, then we need to use a virtual environment first, then apply the ROS environment, ROS environment is Jeffry's task, coordinate with Cristian for help 
Link is:  
https://www.youtube.com/watch?v=OSoMSVry-8E  

Using the Hypersen github for this specific lidar sensor, we are not able to run the demo nor the software on the RPI5 due to the incompatibility being the "libhps3d.so" file used to run the program is formatted for x86-64 Intel/AMD while our RPI5 is aarch64(ARM64) thus we need to find a SDK that is compatible with the ARM64. Same can be said for Jetson Orin Nano for the architecture is ARM64.

To make sure the lidar is connected run:  
```bash
lsusb
```
should display: "ID 0483:5740 STMicroelectronics Virtual COM Port"

## First task- Get offical SDK clone and test
```bash
sudo apt-get update
sudo apt-get install build-essential libusb-1.0-0-dev git cmake
cd ~
git clone https://github.com/DFRobotdl/DFR0728_HPS3D_SDK-HPS3D160_SDK-V1.8.git
cd DFR0728_HPS3D_SDK-HPS3D160_SDK-V1.8/V1.8/Example/HPS3D160-Raspberry-C_Demo/
make
sudo ./app
```
Once above code is implemented, test and see if terminal displays "Distance: 1250 mm | Signal Strength: 2400".  
Check that make works and that whats cloned is for 64 bit not 32 bit, if installed for wrong arch., make the file point towards the arch64 (look into the line where it says cd rasperry demo). If 64 bit arch not available, might need to configure rpi5 to me on 32-bit, not ideal tho.  
### Results
Lidar worked with tested code, had to recompile a new file due to architecture was for pc and not rpi5, thus lidar proves that it works, now need to implement this with the sensors together, need to figure out how to implement this with cameras.

## Second task- Integrate ROS2 for rover to use sensor once lidar is already tested with code functioning as intended

