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
