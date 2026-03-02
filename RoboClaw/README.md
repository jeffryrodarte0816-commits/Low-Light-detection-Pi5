# Roboclaw motor controller
Roboclaw motor controller within the roboclaw_motor_test.py is supposed to move the servos forward, then stop, then reverse, then stop in a sequence.
## Key details from manual used onto script made
1.Default address 0x80 (128) — no need to change unless you set a custom one in Motion Studio  
2.Speed 64/127 = ~50% — pass --speed 32 for ~25% or --speed 96 for ~75%  
3.CRC16 CCITT 0x1021 polynomial, seed 0 — exactly as the manual's Arduino reference  
4.Packet: [Address, Command, Value, CRC16_hi, CRC16_lo] with 0xFF ACK  
## Sequence itself
Motor Directions for Side-to-Side (Skid-Steer)  
Since the front and back wheels on each side are linked to the same motor channel, your wiring should look like this:  
Left Side: Connect both left-side motors in parallel to the M1A and M1B terminals.  
Right Side: Connect both right-side motors in parallel to the M2A and M2B terminals.  
  
Setting the Correct Spin:  

Command: Forward  
Goal: Both sides spin to move the rover forward.  
Fix: If the left side spins backward, swap the wires on M1A/M1B. If the right side spins backward, swap the wires on M2A/M2B.  

Command: Turn Right  
Goal: The robot should rotate clockwise.  
Action: In Mixed Mode, the RoboClaw automatically sends the Left side (M1) forward and the Right side (M2) backward.  

## Dependencie needed
```bash
pip install pyserial 
```
```bash
python3 roboclaw_motor_test.py  
```
//if port not detected, do:
```bash
python3 roboclaw_motor_test.py --port /dev/ttyACM0
```
