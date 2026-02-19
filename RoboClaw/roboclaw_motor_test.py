"""
roboclaw_motor_test.py
======================
Motor test script for RoboClaw 2x15A (Model: IMC412)
Jetson Orin Nano — USB connection

Based on RoboClaw Series User Manual:
  - Default address: 0x80 (128)
  - Packet Serial over USB (Linux auto-detects, no baud rate needed)
  - CRC16 CCITT polynomial 0x1021, seed 0
  - Command 0  : Drive M1 Forward  (0–127)
  - Command 1  : Drive M1 Backward (0–127)
  - Command 4  : Drive M2 Forward  (0–127)
  - Command 5  : Drive M2 Backward (0–127)
  - Packet format: [Address, Command, Value, CRC16_hi, CRC16_lo]
  - ACK response: 0xFF

Test Sequence:
    1. M1 Forward  at ~50% speed for TEST_DURATION seconds
    2. M1 Stop
    3. M1 Backward at ~50% speed for TEST_DURATION seconds
    4. M1 Stop
    5. M2 Forward  at ~50% speed for TEST_DURATION seconds
    6. M2 Stop
    7. M2 Backward at ~50% speed for TEST_DURATION seconds
    8. M2 Stop
    9. Both motors forward simultaneously at ~50% for TEST_DURATION seconds
   10. Both motors stop — done

Usage:
    source /home/sunnysquad/venv/bin/activate
    python3 roboclaw_motor_test.py

    # If port is not auto-detected, pass it manually:
    python3 roboclaw_motor_test.py --port /dev/ttyACM0
"""

import serial
import serial.tools.list_ports
import time
import argparse
import sys

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — adjust if needed
# ─────────────────────────────────────────────────────────────────
ROBOCLAW_ADDRESS = 0x80          # Default address per manual (128)
BAUD_RATE        = 38400         # USB CDC ignores this, but pyserial requires a value
TEST_SPEED       = 64            # ~50% speed (range 0–127 per manual)
TEST_DURATION    = 2.0           # Seconds per test phase
STOP_PAUSE       = 0.5           # Pause between phases
SERIAL_TIMEOUT   = 1.0           # Read timeout in seconds

# ─────────────────────────────────────────────────────────────────
# COMMANDS (from manual)
# ─────────────────────────────────────────────────────────────────
CMD_M1_FORWARD   = 0
CMD_M1_BACKWARD  = 1
CMD_M2_FORWARD   = 4
CMD_M2_BACKWARD  = 5


# ─────────────────────────────────────────────────────────────────
# CRC16 — CCITT polynomial 0x1021, seed 0
# Matches the manual's Arduino reference implementation exactly
# ─────────────────────────────────────────────────────────────────
def crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF   # keep 16-bit
    return crc


# ─────────────────────────────────────────────────────────────────
# PACKET BUILDER & SENDER
# Format: [Address, Command, Value, CRC16_hi, CRC16_lo]
# ─────────────────────────────────────────────────────────────────
def send_command(ser: serial.Serial, address: int, command: int, value: int) -> bool:
    """
    Build and send a packet serial command.
    Returns True if RoboClaw ACKs with 0xFF, False otherwise.
    """
    packet = bytes([address, command, value])
    crc    = crc16(packet)
    crc_hi = (crc >> 8) & 0xFF
    crc_lo = crc & 0xFF
    full_packet = bytes([address, command, value, crc_hi, crc_lo])

    ser.write(full_packet)
    ser.flush()

    # Wait for ACK (0xFF per manual)
    ack = ser.read(1)
    if ack == b'\xff':
        return True
    else:
        print(f"  ⚠️  No ACK received. Got: {ack.hex() if ack else 'nothing'}")
        return False


def stop_m1(ser):
    send_command(ser, ROBOCLAW_ADDRESS, CMD_M1_FORWARD, 0)

def stop_m2(ser):
    send_command(ser, ROBOCLAW_ADDRESS, CMD_M2_FORWARD, 0)

def stop_all(ser):
    stop_m1(ser)
    stop_m2(ser)


# ─────────────────────────────────────────────────────────────────
# AUTO PORT DETECTION
# RoboClaw on Linux shows up as /dev/ttyACM* or /dev/ttyUSB*
# ─────────────────────────────────────────────────────────────────
def find_roboclaw_port() -> str | None:
    print("🔍 Scanning for RoboClaw USB port...")
    ports = serial.tools.list_ports.comports()
    candidates = []

    for port in ports:
        desc = (port.description or "").lower()
        mfr  = (port.manufacturer or "").lower()
        # RoboClaw shows as CDC ACM or USB serial
        if any(kw in desc for kw in ["roboclaw", "basicmicro", "cdc", "acm"]) or \
           any(kw in mfr  for kw in ["roboclaw", "basicmicro"]):
            candidates.append(port.device)
            print(f"  ✅ Found: {port.device} — {port.description}")

    # Fallback: grab any ACM/USB serial port
    if not candidates:
        for port in ports:
            if "ttyACM" in port.device or "ttyUSB" in port.device:
                candidates.append(port.device)
                print(f"  📌 Candidate: {port.device} — {port.description}")

    if candidates:
        print(f"  → Using: {candidates[0]}")
        return candidates[0]

    print("  ❌ No RoboClaw port found.")
    print("  Available ports:")
    for p in ports:
        print(f"    {p.device}: {p.description}")
    return None


# ─────────────────────────────────────────────────────────────────
# TEST SEQUENCE
# ─────────────────────────────────────────────────────────────────
def run_test(ser: serial.Serial):
    print(f"\n{'─'*50}")
    print(f"  RoboClaw Motor Test")
    print(f"  Address  : 0x{ROBOCLAW_ADDRESS:02X} ({ROBOCLAW_ADDRESS})")
    print(f"  Speed    : {TEST_SPEED}/127 (~{TEST_SPEED/127*100:.0f}%)")
    print(f"  Duration : {TEST_DURATION}s per phase")
    print(f"{'─'*50}\n")

    # Safety: ensure stopped before starting
    print("⛔ Ensuring motors stopped before test...")
    stop_all(ser)
    time.sleep(STOP_PAUSE)

    tests = [
        ("M1 Forward",          CMD_M1_FORWARD,  TEST_SPEED),
        ("M1 Stop",             CMD_M1_FORWARD,  0),
        ("M1 Backward",         CMD_M1_BACKWARD, TEST_SPEED),
        ("M1 Stop",             CMD_M1_FORWARD,  0),
        ("M2 Forward",          CMD_M2_FORWARD,  TEST_SPEED),
        ("M2 Stop",             CMD_M2_FORWARD,  0),
        ("M2 Backward",         CMD_M2_BACKWARD, TEST_SPEED),
        ("M2 Stop",             CMD_M2_FORWARD,  0),
    ]

    for label, cmd, val in tests:
        is_stop = val == 0
        duration = STOP_PAUSE if is_stop else TEST_DURATION

        print(f"▶  {label}{'...' if not is_stop else ''}")
        ack = send_command(ser, ROBOCLAW_ADDRESS, cmd, val)
        if ack:
            print(f"   ✅ ACK received")
        time.sleep(duration)

    # Both motors forward simultaneously
    print("▶  Both M1 + M2 Forward simultaneously...")
    ack1 = send_command(ser, ROBOCLAW_ADDRESS, CMD_M1_FORWARD, TEST_SPEED)
    ack2 = send_command(ser, ROBOCLAW_ADDRESS, CMD_M2_FORWARD, TEST_SPEED)
    if ack1 and ack2:
        print("   ✅ Both ACKs received")
    time.sleep(TEST_DURATION)

    print("⛔ Stopping all motors...")
    stop_all(ser)
    time.sleep(STOP_PAUSE)

    print(f"\n{'─'*50}")
    print("  ✅ Motor test complete. All motors stopped.")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RoboClaw 2x15A Motor Test")
    parser.add_argument("--port",    type=str, default=None,
                        help="Serial port (e.g. /dev/ttyACM0). Auto-detected if not set.")
    parser.add_argument("--speed",   type=int, default=TEST_SPEED,
                        help=f"Speed value 0–127 (default: {TEST_SPEED} = ~50%%)")
    parser.add_argument("--duration",type=float, default=TEST_DURATION,
                        help=f"Seconds per test phase (default: {TEST_DURATION})")
    args = parser.parse_args()

    # Allow CLI overrides
    global TEST_SPEED, TEST_DURATION
    TEST_SPEED    = max(0, min(127, args.speed))
    TEST_DURATION = args.duration

    # Port discovery
    port = args.port or find_roboclaw_port()
    if not port:
        print("\n❌ No port found. Try: python3 roboclaw_motor_test.py --port /dev/ttyACM0")
        print("   Also check: ls /dev/ttyACM* /dev/ttyUSB*")
        sys.exit(1)

    # Open serial
    print(f"\n🔌 Connecting to RoboClaw on {port}...")
    try:
        ser = serial.Serial(
            port      = port,
            baudrate  = BAUD_RATE,       # USB CDC ignores this per manual
            timeout   = SERIAL_TIMEOUT,
            bytesize  = serial.EIGHTBITS,
            parity    = serial.PARITY_NONE,
            stopbits  = serial.STOPBITS_ONE
        )
        print(f"   ✅ Connected to {port}")
    except serial.SerialException as e:
        print(f"\n❌ Could not open {port}: {e}")
        print("   Try: sudo chmod 666 /dev/ttyACM0")
        print("   Or add user to dialout group: sudo usermod -aG dialout $USER")
        sys.exit(1)

    try:
        run_test(ser)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted — stopping all motors...")
        stop_all(ser)
        print("✅ Motors stopped safely.")
    finally:
        stop_all(ser)   # Always stop on exit
        ser.close()
        print("🔌 Serial port closed.")


if __name__ == "__main__":
    main()
