import serial
import csv
import sys
import os
import time
import numpy as np
from datetime import datetime
from collections import deque

# ── Import shared sensor config from project root ─────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sensor_config as sc

# ── Configuration ─────────────────────────────────────────────────────────────
SERIAL_PORT    = "COM5"   # Adjust when needed
BAUD_RATE      = 115200
GESTURE_LABEL  = "A"      # Change with every recording
OUTPUT_DIR     = "data/raw"
OUTPUT_FILE    = f"data/raw/gesture_{GESTURE_LABEL}.csv"

"""
First Checkoff:  1, 2, 3, OK, W
Second Checkoff: A, B, C, D, E, F, I, L, O, Y
Third Checkoff:  J, M, N, P, S, T, U, V, X, Z
Final:           R, K, Q, G, H
"""

# ── Smoothing Settings ────────────────────────────────────────────────────────
ENABLE_SMOOTHING = True
SMOOTHING_WINDOW = 5      # Number of samples to average

# ── Load packet format from shared config ─────────────────────────────────────
active_fields   = sc.get_active_fields()          # includes timestamp if enabled
sensor_fields   = sc.get_active_sensor_fields()   # excludes timestamp
EXPECTED_LEN    = sc.get_expected_len()           # total values per packet
N_SENSORS       = len(sensor_fields)              # values going into CSV / smoothing

# ── Print startup info ────────────────────────────────────────────────────────
print("=" * 50)
print("Enhanced Data Logger - Team 7 Senior Design")
print("=" * 50)
print(f"\nConfiguration:")
print(f"  Port:           {SERIAL_PORT}")
print(f"  Baud Rate:      {BAUD_RATE}")
print(f"  Gesture Label:  '{GESTURE_LABEL}'")
print(f"  Output File:    {OUTPUT_FILE}")
print(f"  Smoothing:      {'Enabled' if ENABLE_SMOOTHING else 'Disabled'}")
if ENABLE_SMOOTHING:
    print(f"  Smooth Window:  {SMOOTHING_WINDOW} samples")
print(f"\nPacket format ({EXPECTED_LEN} values per line):")
print(f"  {', '.join(active_fields)}")
print(f"\nCSV columns: {', '.join(sensor_fields + ['label'])}")
print(f"\nPress Ctrl+C to stop recording\n")

# ── Connect ───────────────────────────────────────────────────────────────────
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Failed to connect to {SERIAL_PORT}: {e}")
    exit(1)

# ── Init smoothing buffers (one per sensor field) ─────────────────────────────
sensor_buffers = [deque(maxlen=SMOOTHING_WINDOW) for _ in range(N_SENSORS)]

# ── Recording loop ────────────────────────────────────────────────────────────
sample_count = 0
error_count  = 0
start_time   = time.time()

# Append to existing file or create new one with header
file_exists = os.path.isfile(OUTPUT_FILE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, mode="a", newline="") as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(sc.get_csv_header(include_label=True))

    try:
        while True:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode("utf-8").strip()

                    # Parse using shared config
                    parsed = sc.parse_packet(line)
                    if parsed is None:
                        error_count += 1
                        got = len(line.split(","))
                        print(f"Warning: Expected {EXPECTED_LEN} values, got {got} — '{line[:60]}'")
                        continue

                    # Validate ranges
                    ok, warnings = sc.validate_packet(parsed)
                    if not ok:
                        for w in warnings:
                            print(f"WARNING: {w}")
                        error_count += 1
                        continue

                    # Extract sensor values in field order (no timestamp)
                    sensor_values = [parsed[f] for f in sensor_fields]

                    # Apply smoothing if enabled
                    if ENABLE_SMOOTHING:
                        final_values = []
                        for i, val in enumerate(sensor_values):
                            sensor_buffers[i].append(val)
                            final_values.append(np.mean(sensor_buffers[i]))
                    else:
                        final_values = sensor_values

                    # Write row: sensor values + label
                    writer.writerow([*final_values, GESTURE_LABEL])

                    # Progress print every 10 samples
                    sample_count += 1
                    if sample_count % 10 == 0:
                        elapsed = time.time() - start_time
                        rate    = sample_count / elapsed if elapsed > 0 else 0

                        # Flex summary (first 5 sensor fields that are flex)
                        flex_vals = [final_values[i]
                                     for i, f in enumerate(sensor_fields)
                                     if f.startswith("flex_")]
                        flex_str  = ", ".join(f"{v:.1f}" for v in flex_vals)

                        # IMU magnitude if available
                        accel_fields = [i for i, f in enumerate(sensor_fields)
                                        if f.startswith("accel_")]
                        gyro_fields  = [i for i, f in enumerate(sensor_fields)
                                        if f.startswith("gyro_")]

                        imu_str = ""
                        if accel_fields:
                            accel_mag = np.sqrt(sum(final_values[i]**2 for i in accel_fields))
                            imu_str += f" | Accel: {accel_mag:.0f}"
                        if gyro_fields:
                            gyro_mag = np.sqrt(sum(final_values[i]**2 for i in gyro_fields))
                            imu_str += f" | Gyro: {gyro_mag:.0f}"

                        print(f"Sample {sample_count} | Rate: {rate:.1f} Hz"
                              f" | Flex: [{flex_str}]{imu_str}")

                except UnicodeDecodeError:
                    error_count += 1
                    print("WARNING: Failed to decode serial data")
                except Exception as e:
                    error_count += 1
                    print(f"WARNING: {e}")

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Recording stopped by user")
        print("=" * 50)

# ── Summary ───────────────────────────────────────────────────────────────────
elapsed = time.time() - start_time
print(f"\nRecording Summary:")
print(f"  Total samples:  {sample_count}")
print(f"  Errors:         {error_count}")
print(f"  Duration:       {elapsed:.1f} seconds")
if elapsed > 0:
    print(f"  Average rate:   {sample_count / elapsed:.1f} Hz")
print(f"  Output file:    {OUTPUT_FILE}")
print(f"\nData saved successfully!\n")

ser.close()