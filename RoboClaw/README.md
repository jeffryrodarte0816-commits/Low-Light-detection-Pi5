# Roboclaw motor controller
Roboclaw motor controller within the roboclaw_motor_test.py is supposed to move the servos forward, then stop, then reverse, then stop in a sequence.
## Key details from manual used onto script made
1.Default address 0x80 (128) — no need to change unless you set a custom one in Motion Studio  
2.Speed 64/127 = ~50% — pass --speed 32 for ~25% or --speed 96 for ~75%  
3.CRC16 CCITT 0x1021 polynomial, seed 0 — exactly as the manual's Arduino reference  
4.Packet: [Address, Command, Value, CRC16_hi, CRC16_lo] with 0xFF ACK  
## Sequence itself
Detail is 9 phases total, 2 seconds each at ~50% speed:
1.M1 forward → stop → backward → stop  
2.M2 forward → stop → backward → stop  
3.Both M1 + M2 forward simultaneously → stop  

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
