# Philip Sherman
# Team 7 - Senior Design
# 2/2/2026

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
MIN_SENSOR_VALUE = 0
MAX_SENSOR_VALUE = 100
EXPECTED_SENSORS = 5

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
    writer.writerow(["timestamp", "thumb", "index", "middle", "ring", "pinky", "label"])
    
    try:
        while True:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode("utf-8").strip()
                    values = line.split(",")

                    # Validate data
                    if len(values) != EXPECTED_SENSORS:
                        error_count += 1
                        print(f"⚠ Warning: Expected {EXPECTED_SENSORS} values, got {len(values)}")
                        continue
                    
                    # Convert to float and validate range
                    sensor_values = []
                    valid = True
                    for i, v in enumerate(values):
                        try:
                            val = float(v)
                            if not (MIN_SENSOR_VALUE <= val <= MAX_SENSOR_VALUE):
                                print(f"WARNING: Sensor {i} value {val} out of range")
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
                        print(f"Sample {sample_count} | Rate: {rate:.1f} Hz | Values: [{', '.join([f'{v:.1f}' for v in final_values])}]")
                
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
print(f"\n✓ Data saved successfully!\n")

ser.close()