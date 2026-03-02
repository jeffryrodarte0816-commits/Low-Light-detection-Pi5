"""
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
"""

import serial
import time

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
# On RPi5, GPIO UART is typically /dev/ttyAMA0
PORT = "/dev/ttyAMA0" 
BAUD_RATE = 38400
ADDRESS = 0x80

# Mixed Mode Commands
CMD_FWD = 8
CMD_REV = 9
CMD_RIGHT = 10
CMD_LEFT = 11

def crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc

def send_move(ser, cmd, val):
    packet = bytes([ADDRESS, cmd, val])
    crc = crc16(packet)
    full_packet = packet + bytes([(crc >> 8) & 0xFF, crc & 0xFF])
    ser.write(full_packet)
    ser.flush()
    return ser.read(1) == b'\xff'

def main():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1.0)
        print(f"Connected to RoboClaw on {PORT}")
        
        print("Moving Forward...")
        send_move(ser, CMD_FWD, 64) # 50% speed
        time.sleep(2)
        
        print("Stopping...")
        send_move(ser, CMD_FWD, 0)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    main()
