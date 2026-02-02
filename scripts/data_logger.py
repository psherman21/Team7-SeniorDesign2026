# Philip Sherman
# Team 7 - Senior Design
# 2/2/2025

import serial
import csv
import time
from datetime import datetime

SERIAL_PORT = "COM3" # Adjust when needed
BAUD_RATE = 115200
GESTURE_LABEL = "A" # Change with every recording
OUTPUT_FILE = f"data/raw/gesture_{GESTURE_LABEL}.csv"

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

print(f"Recording gesture '{GESTURE_LABEL}'")
print("Press Ctrl+C to stop recording\n")

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