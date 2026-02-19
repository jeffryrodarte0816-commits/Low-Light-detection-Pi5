# Installing ultralytics for yolo on jetson orin nano
OS Version: Jetpack 6.2.1  

Verify version of python, what version of CUDA the OS supports, and how to install pytorch first
```bash
python --version
```
```bash
nvidia-smi
```
At the moment: python-3.10.12, CUDA version: 12.6  
Now go to website for quickstart for ultralytics and place parameters needed for installation, install "pip" since we do have CUDA for the jetson.  
```bash
sudo apt install python3-pip
```
Next Install the virtual environment, which we need to activate to then install the ultralytics within the venv:
```bash
sudo apt install -y python3-venv
python3 -m venv venv
```
Now that we made folder called "venv" with the environment within, we open the venv:
```bash
source venv/bin/activate
```
To deactivate venv once done whatever task:
```bash
deactivate
```

Once installed the pytorch and torchvision(gave a hassle),then install ultralytics:
```bash
pip install ultralytics
```

