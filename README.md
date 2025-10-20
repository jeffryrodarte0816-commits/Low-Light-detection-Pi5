Purpose is to run on the Raspberry PI 5 to run Yolov11 using Python to multi-detect objects. We are checking the latest version of python and downloading ultralytics. Problem arrises when dowloading ultralytics(YOLO) .
Thus we remove Trixie(unstable version) to be replaced by a stable version called Bookworm. 
Once stable environmet called Bookworm is in the PI 5 by using grep to access and read "Suites" where it should show bookworm
Then install the virtual environment using python, but we get an error saying outdated packages
