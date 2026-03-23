import serial
import csv
import time
import numpy as np
from datetime import datetime
from collections import deque

# Configuration
SERIAL_PORT = "COM5" # Adjust when needed
# TO DO -- ADD Bluetooth connection
BAUD_RATE = 115200
GESTURE_LABEL = "A" # Change with every recording
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = f"data/raw/gesture_{GESTURE_LABEL}.csv"
"""
First Checkoff: 1, 2, 3, OK, W
Second Checkoff: A, B, C, D, E, F, I, L, O, Y
Third Checkoff: J, M, N, P, S, T, U, V, X, Z
Final: R, K, Q, G, H

"""

# Smoothing Settings
ENABLE_SMOOTHING = True
SMOOTHING_WINDOW = 5   # Number of samples being averaged

# Validation Settings
MIN_FLEX_VALUE = 0
MAX_FLEX_VALUE = 100
MIN_ACCEL_VALUE = -20000  # Typical range for accelerometer
MAX_ACCEL_VALUE = 20000
MIN_GYRO_VALUE = -20000   # Typical range for gyroscope
MAX_GYRO_VALUE = 20000
EXPECTED_SENSORS = 11  # 5 flex + 3 accel + 3 gyro

# ===== Initialize =====
print("="*50)
print("Enhanced Data Logger - Team 7 Senior Design")
print("="*50)
print(f"\nConfiguration:")
print(f"  Port: {SERIAL_PORT}")
print(f"  Baud Rate: {BAUD_RATE}")
print(f"  Gesture Label: '{GESTURE_LABEL}'")
print(f"  Output File: {OUTPUT_FILE}")
print(f"  Smoothing: {'Enabled' if ENABLE_SMOOTHING else 'Disabled'}")
if ENABLE_SMOOTHING:
    print(f"  Smoothing Window: {SMOOTHING_WINDOW} samples")
print(f"\nPress Ctrl+C to stop recording\n")

# Connect
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Failed to connect to {SERIAL_PORT}")
    print(f"Error: {e}")
    exit(1)

# Init Smoothing Buffer
sensor_buffers = [deque(maxlen=SMOOTHING_WINDOW) for _ in range(EXPECTED_SENSORS)]

# Recording Loop
sample_count = 0
error_count = 0
start_time = time.time()

with open(OUTPUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "flex_1", "flex_2", "flex_3", "flex_4", "flex_5",
                     "accel_x", "accel_y", "accel_z",
                     "gyro_x", "gyro_y", "gyro_z", "label"])
    
    try:
        while True:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode("utf-8").strip()
                    values = line.split(",")

                    # Validate data
                    if len(values) != EXPECTED_SENSORS:
                        error_count += 1
                        print(f"Warning: Expected {EXPECTED_SENSORS} values, got {len(values)}")
                        continue
                    
                    # Convert to float and validate range
                    sensor_values = []
                    valid = True
                    for i, v in enumerate(values):
                        try:
                            val = float(v)
                            if i < 5:  # Flex sensors (0-4)
                                if not (MIN_FLEX_VALUE <= val <= MAX_FLEX_VALUE):
                                    print(f"WARNING: Flex sensor {i} value {val} out of range")
                                    valid = False
                                    break
                            elif i < 8:  # Accelerometer (5-7)
                                if not (MIN_ACCEL_VALUE <= val <= MAX_ACCEL_VALUE):
                                    print(f"WARNING: Accelerometer {i-5} value {val} out of range")
                                    valid = False
                                    break
                            else:  # Gyroscope (8-10)
                                if not (MIN_GYRO_VALUE <= val <= MAX_GYRO_VALUE):
                                    print(f"WARNING: Gyroscope {i-8} value {val} out of range")
                                    valid = False
                                    break

                            sensor_values.append(val)
                        except ValueError:
                            print(f"WARNING: Invalid value '{v}' for sensor {i}")
                            valid = False
                            break
                    
                    if not valid:
                        error_count += 1
                        continue
                    
                    # Apply smoothing if enabled
                    if ENABLE_SMOOTHING:
                        smoothed_values = []
                        for i, val in enumerate(sensor_values):
                            sensor_buffers[i].append(val)
                            smoothed_values.append(np.mean(sensor_buffers[i]))
                        final_values = smoothed_values
                    else:
                        final_values = sensor_values

                    # Write to CSV
                    timestamp = datetime.now().isoformat()
                    writer.writerow([timestamp, *final_values, GESTURE_LABEL])
                    
                    # Print progress
                    sample_count += 1
                    if sample_count % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = sample_count / elapsed if elapsed > 0 else 0

                        flex_str = ', '.join([f'{v:.1f}' for v in final_values[:5]])
                        accel_mag = np.sqrt(sum([v**2 for v in final_values[5:8]]))
                        gyro_mag = np.sqrt(sum([v**2 for v in final_values[8:11]]))
                        print(f"Sample {sample_count} | Rate: {rate:.1f} Hz | Flex: [{flex_str}] | Accel: {accel_mag:.0f} | Gyro: {gyro_mag:.0f}")
                except UnicodeDecodeError:
                    error_count += 1
                    print("WARNING: Failed to decode serial data")
                except Exception as e:
                    error_count += 1
                    print(f"WARNING: {e}")

    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("Recording stopped by user")
        print("="*50)

# Summary
elapsed = time.time() - start_time
print(f"\nRecording Summary:")
print(f"  Total samples: {sample_count}")
print(f"  Errors: {error_count}")
print(f"  Duration: {elapsed:.1f} seconds")
if elapsed > 0:
    print(f"  Average rate: {sample_count/elapsed:.1f} Hz")
print(f"  Output file: {OUTPUT_FILE}")
print(f"\n Data saved successfully!\n")

ser.close()