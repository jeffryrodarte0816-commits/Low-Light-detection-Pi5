# Sensor Integration (Starting out with tc001 and day/night arducam)
https://github.com/leswright1977/PyThermalCamera/blob/main/README.md  
Use the Homography method to integrate both sensors. Need opencv and numpy.
First connect the tc001, then verify if dev/video0, then connect arducam and verify if dev/video2.  
We ran the "test_arducam.py" to verify the arducam is video2 which it was.  
To verify the imports used within a specific script:(Example below)
```bash
python3 -c "import cv2, numpy, argparse, time, io, sys; print('✅ Success! All libraries are installed.')"
```
