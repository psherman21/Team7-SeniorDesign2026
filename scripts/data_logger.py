# Philip Sherman
# Team 7 - Senior Design
# 2/2/2025

import serial
import csv
import time
import numpy as np
from datetime import datetime
from collections import deque

# Configuration
SERIAL_PORT = "COM3" # Adjust when needed
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
            line = ser.readline().decode("utf-8").strip()
            values = line.split(",")

            if len(values) != 5:
                continue

            timestamp = datetime.now().isoformat()
            writer.writerow([timestamp, *values, GESTURE_LABEL])
            print(values)

    except KeyboardInterrupt:
        print("\nRecording stopped.")

ser.close()