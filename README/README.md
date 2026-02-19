# Purpose
Raspberry PI 5 to run Yolov11 using Python to multi-detect objects. Running the YoloV8 under the ultralytics.
# Progress
1. Thus we remove Trixie(unstable version) to be replaced by a stable version called Bookworm. 
2. Once stable environmet called Bookworm is in the PI 5 by using grep to access and read "Suites" where it should show bookworm
3. Then have the Pi 5 access the virtual environment using python, should get no error
4. Then install virtual environment named "venv". 
5. Do pip install setuptools numpy Cython , it should install everything
6. Do pip install request, what this does it basically install the pytorch which is needed to install ultralytics
7. From here down use the install ultralytics website:
8. Now you clone the ultralytic depository and install whats needed
9. What we need now is to use CLI to start using the YOLO by inputting specific commands
10. The CLI is still in proccess but progrss made

# Possible Methods for labeling each Image 
We can try to install the "labelImg" on another computer to automatically generate the txt file for each image. When adding images to the yolo, each image is supposed to have own text file. Thus we can install the labelImg on a computer then add the txt file along with it's corresponding pictures to the yolo.
