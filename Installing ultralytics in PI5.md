This is to make the version from a unstable(trixie) to stable(bookworm)
```bash
sudo sed -i 's/trixie/bookworm/g' etc/apt/sources.list.d/*.sources
grep -R "Suites" /etc/apt/sources.list.d/
```

```bash
sudo apt install python3-pip
python3 -m venv venv
source venv/bin/activate
pip3 install setuptools numpy Cython
pip3 install requests
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

```bash
sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

workon pyt

pip3 install setuptools numpy Cython
pip3 install requests
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install ultralytics

 git clone https://github.com/ultralytics/ultralytics
 cd ultralytics
```
