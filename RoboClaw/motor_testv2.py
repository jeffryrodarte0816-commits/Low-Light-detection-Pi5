"""
M1 moves backwards and then immediatly after the M2 moves backwards
"""

import serial
import time

PORT = "/dev/ttyAMA0"
BAUD_RATE = 38400
ADDRESS = 0x80

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
        print("🚀 Synchronizing M1 and M2 BACKWARD...")

        # Send M1 backward, then M2 immediately after
        # Use a slightly lower speed to prevent a battery voltage spike
        speed = 40
        
        success_m1 = send_move(ser, 1, speed) # M1 Backward
        time.sleep(0.01)                      # Tiny 10ms gap for the processor
        success_m2 = send_move(ser, 5, speed) # M2 Backward

        if success_m1 and success_m2:
            print("✅ Both sides engaged.")
        else:
            print("⚠️ One or both sides failed to ACK.")

        time.sleep(3) # Drive for 3 seconds

        print("⏹ Stopping...")
        send_move(ser, 1, 0)
        time.sleep(0.01)
        send_move(ser, 5, 0)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals(): ser.close()

if __name__ == "__main__":
    main()
