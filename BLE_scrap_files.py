import tkinter as tk
from tkinter import ttk, messagebox
import serial
import threading
import time
import numpy as np
import joblib
from collections import deque
import ctypes
import json
import asyncio
from bleak import BleakClient, BleakScanner

#CHAR_UUID = "abcd1234-1234-1234-1234-abcdef123456"
#MAC_ADDRESS = "f4:2d:c9:72:5d:64"

class GloveUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Team 7 Senior Design Glove Demo")
        self.geometry("1450x850")

        # Serial/ML variables
        self.ser = None
        self.ble_client = None
        self.ble_loop = None
        self.ble_thread = None
        self.latest_data = None

def connect_glove(self):
        """Connect to ESP32 via BLE"""
        
        async def ble_connect():
            try:
                self.status_label.config(text="Status: Scanning...", fg="orange")
                
                devices = await BleakScanner.discover(timeout=5.0)
                esp_device = None
                
                for d in devices:
                    if d.name and "ESP32" in d.name:  # Change if your device name differs
                        esp_device = d
                        break
                
                if not esp_device:
                    raise Exception("ESP32 not found")
                
                self.status_label.config(text="Status: Connecting...", fg="orange")
                
                self.ble_client = BleakClient(esp_device.address)
                await self.ble_client.connect()
                
                await self.ble_client.start_notify(CHAR_UUID, self.notification_handler)
                
                self.status_label.config(text="Status: Connected (BLE)", fg=CONFIDENCE_HIGH)
                messagebox.showinfo("Success", "Connected to glove via BLE")
            
            except Exception as e:
                self.status_label.config(text="Status: Connection Failed", fg=CONFIDENCE_LOW)
                messagebox.showerror("Error", str(e))

        # Run BLE in separate thread (because Tkinter + asyncio conflict)
        def run_loop():
            self.ble_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.ble_loop)
            self.ble_loop.run_until_complete(ble_connect())
            self.ble_loop.run_forever()

        self.ble_thread = threading.Thread(target=run_loop, daemon=True)
        self.ble_thread.start()

def recognition_loop(self):
        """Main recognition loop running in separate thread"""
        while self.is_recognizing:
            if self.latest_data:
                    values = self.latest_data
            if len(values) == 5:
                sensor_values = [float(v) for v in values]
                    
                # Apply hand size calibration
                calibrated = self.apply_calibration(sensor_values)
                
                # Update sensor display
                self.update_sensor_display(calibrated)
                
                # Add to buffer for smoothing
                self.sensor_buffer.append(calibrated)
                
                # Only predict if we have enough samples
                if len(self.sensor_buffer) >= self.smoothing_window.get():
                    # Average the buffer
                    smoothed = np.mean(list(self.sensor_buffer)[-self.smoothing_window.get():], axis=0)
                    
                    # Make prediction
                    self.predict_gesture(smoothed)
            
        time.sleep(0.1)  # 10Hz update rate