Purpose is to run on the Raspberry PI 5 to run Yolov11 using Python to multi-detect objects. We are checking the latest version of python and downloading ultralytics. Problem arrises when dowloading ultralytics(YOLO) .
Thus we remove Trixie(unstable version) to be replaced by a stable version called Bookworm. 
Once stable environmet called Bookworm is in the PI 5 by using grep to access and read "Suites" where it should show bookworm
Then have the Pi 5 access the virtual environment using python, should get no error
Then install virtual environment named "venv". 
Do pip install setuptools numpy Cython , it should install everything
Do pip install request, what this does it basically install the pytorch which is needed to install ultralytics
From here down use the install ultra;ytics website:
Now you clone the ultralytic depository and install whats needed
What we need now is to use CLI to start using the YOLO by inputting specific commands
The CLI is still in proccess but progrss made
