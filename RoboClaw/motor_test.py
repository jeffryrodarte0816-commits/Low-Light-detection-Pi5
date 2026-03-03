"""
motor_test.py

Since the front and back wheels on each side are linked to the same motor channel, your wiring should look like this:

Left Side: Connect both left-side motors in parallel to the M1A and M1B terminals.

Right Side: Connect both right-side motors in parallel to the M2A and M2B terminals.

Setting the Correct Spin:

m1 forward and m2 forward
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
        
        # TEST 1: LEFT SIDE ONLY
        print("--- Testing LEFT SIDE (M1) ---")
        print("Moving M1 Forward...")
        send_move(ser, 0, 40) # Command 0 = M1 Forward
        time.sleep(2)
        send_move(ser, 0, 0)  # Stop
        print("Check: Did both left wheels move forward?")
        time.sleep(1)

        # TEST 2: RIGHT SIDE ONLY
        print("\n--- Testing RIGHT SIDE (M2) ---")
        print("Moving M2 Forward...")
        send_move(ser, 4, 40) # Command 4 = M2 Forward
        time.sleep(2)
        send_move(ser, 4, 0)  # Stop
        print("Check: Did both right wheels move forward?")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals(): ser.close()

if __name__ == "__main__":
    main()
