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
#import asyncio
#from bleak import BleakClient, BleakScanner

# Color Presets
PRIMARY_COLOR = "#001F54"
SECONDARY_COLOR = "#ADD8E6"
BG_COLOR = "#F0F8FF"
CONFIDENCE_HIGH = "#4CAF50"
CONFIDENCE_MED = "#FFC107"
CONFIDENCE_LOW = "#F44336"

#CHAR_UUID = "abcd1234-1234-1234-1234-abcdef123456" # need to find
#MAC_ADDRESS = "f4:2d:c9:72:5d:64"

class GloveUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Team 7 Senior Design Glove Demo")
        self.geometry("1450x850")
        self.configure(bg=BG_COLOR)

        # Serial/ML variables
        self.ser = None
        #self.ble_client = None
        #self.ble_loop = None
        #self.ble_thread = None
        #self.latest_data = None
        self.model = None
        self.scaler = None
        self.is_recognizing = False
        self.sensor_buffer = deque(maxlen=10)
        self.recorded_data = []
        self.confusion_matrix = None

        # Calibraton values
        self.sensor_min = [0, 0, 0, 0, 0]
        self.sensor_max = [100, 100, 100, 100, 100]

        self.setup_ui()

    def setup_ui(self):
        ctypes.windll.shcore.SetProcessDpiAwareness(True)
        # ===== Title =====
        title_label = tk.Label(self, text="Team 7 Senior Design Project Demo",
                               font=("Arial", 18, "bold"),
                               bg=BG_COLOR, fg=PRIMARY_COLOR)
        title_label.pack(pady=1)
        
        # ===== Main Container =====
        main_container = tk.Frame(self, bg=BG_COLOR)
        main_container.pack(fill="both", expand=True, padx=5, pady=2)
        
        # ===== LEFT COLUMN - Controls and Preferences =====
        left_column = tk.Frame(main_container, bg=BG_COLOR)
        left_column.pack(side="left", fill="y", padx=10)
        left_column.config(width=250)
        
        # Connection Control
        connection_frame = tk.LabelFrame(left_column, text="Connection", 
                                        bg=BG_COLOR, fg=PRIMARY_COLOR,
                                        font=("Arial", 11, "bold"), padx=8, pady=2)
        connection_frame.pack(fill="x", pady=2)
        
        self.status_label = tk.Label(connection_frame, text="Status: Disconnected",
                                     font=("Arial", 9, "bold"), bg=BG_COLOR, fg="red")
        self.status_label.pack(pady=2)
        
        tk.Label(connection_frame, text="COM Port:", font=("Arial", 9),
                bg=BG_COLOR, fg=PRIMARY_COLOR).pack()
        self.port_entry = tk.Entry(connection_frame, font=("Arial", 9))
        self.port_entry.insert(0, "COM5")
        self.port_entry.pack(pady=2)
        
        tk.Button(connection_frame, text="Connect", bg=PRIMARY_COLOR, fg="white",
                 command=self.connect_glove, font=("Arial", 9, "bold")).pack(side="left",padx=2)
        tk.Button(connection_frame, text="Calibrate", bg=PRIMARY_COLOR, fg="white",
                 command=self.calibrate_sensors, font=("Arial", 9)).pack(side="left",padx=2)
        
        # Recognition Control
        recognition_frame = tk.LabelFrame(left_column, text="Recognition Control",
                                         bg=BG_COLOR, fg=PRIMARY_COLOR,
                                         font=("Arial", 11, "bold"), padx=8, pady=2)
        recognition_frame.pack(fill="x", pady=2)
        
        tk.Button(recognition_frame, text="Load Model", bg=PRIMARY_COLOR, fg="white",
                 command=self.load_model, font=("Arial", 9, "bold")).pack(pady=2)
        tk.Button(recognition_frame, text="Start Recognition", bg=CONFIDENCE_HIGH, fg="white",
                 command=self.start_recognition, font=("Arial", 9, "bold")).pack(pady=2)
        tk.Button(recognition_frame, text="Stop Recognition", bg=CONFIDENCE_LOW, fg="white",
                 command=self.stop_recognition, font=("Arial", 9, "bold")).pack(pady=2)
        
        # Preferences Section
        pref_frame = tk.LabelFrame(left_column, text="Preferences", bg=BG_COLOR, fg=PRIMARY_COLOR,
                                   font=("Arial", 11, "bold"), padx=8, pady=2)
        pref_frame.pack(fill="x", pady=2)
        
        # Hand Size (for scaling sensor values)
        tk.Label(pref_frame, text="Hand Size Calibration", font=("Arial", 9, "bold"),
                bg=BG_COLOR, fg=PRIMARY_COLOR).pack(anchor="w")
        self.hand_size = tk.DoubleVar(value=1.0)
        hand_scale = tk.Scale(pref_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
                             variable=self.hand_size, bg=SECONDARY_COLOR, fg=PRIMARY_COLOR,
                             highlightbackground=BG_COLOR, length=120)
        hand_scale.pack(fill='x', pady=2)
        
        # KNN Neighbors (changed from "sensitivity")
        tk.Label(pref_frame, text="KNN Neighbors (k)", font=("Arial", 9, "bold"),
                bg=BG_COLOR, fg=PRIMARY_COLOR).pack(anchor="w")
        self.k_neighbors = tk.IntVar(value=5)
        k_scale = tk.Scale(pref_frame, from_=1, to=15, orient=tk.HORIZONTAL,
                          variable=self.k_neighbors, bg=SECONDARY_COLOR, fg=PRIMARY_COLOR,
                          highlightbackground=BG_COLOR)
        k_scale.pack(fill='x', pady=2)
        
        # Confidence Threshold
        tk.Label(pref_frame, text="Confidence Threshold (%)", font=("Arial", 9, "bold"),
                bg=BG_COLOR, fg=PRIMARY_COLOR).pack(anchor="w")
        self.confidence_threshold = tk.DoubleVar(value=60.0)
        conf_scale = tk.Scale(pref_frame, from_=0, to=100, resolution=5, orient=tk.HORIZONTAL,
                             variable=self.confidence_threshold, bg=SECONDARY_COLOR, 
                             fg=PRIMARY_COLOR, highlightbackground=BG_COLOR, length=120)
        conf_scale.pack(fill='x', pady=2)
        
        # Smoothing Window
        tk.Label(pref_frame, text="Sensor Smoothing", font=("Arial", 9, "bold"),
                bg=BG_COLOR, fg=PRIMARY_COLOR).pack(anchor="w")
        self.smoothing_window = tk.IntVar(value=5)
        smooth_scale = tk.Scale(pref_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                               variable=self.smoothing_window, bg=SECONDARY_COLOR,
                               fg=PRIMARY_COLOR, highlightbackground=BG_COLOR, length=120)
        smooth_scale.pack(fill='x', pady=2)
        
        # Recording Section
        recording_frame = tk.LabelFrame(left_column, text="Data Collection",
                                       bg=BG_COLOR, fg=PRIMARY_COLOR,
                                       font=("Arial", 11, "bold"), padx=8, pady=2)
        recording_frame.pack(fill="x", pady=2)
        
        tk.Label(recording_frame, text="Gesture Label:", font=("Arial", 9),
                bg=BG_COLOR, fg=PRIMARY_COLOR).pack()
        self.gesture_label_entry = tk.Entry(recording_frame, font=("Arial", 9), width=15)
        self.gesture_label_entry.insert(0, "A")
        self.gesture_label_entry.pack(pady=2)
        
        tk.Label(recording_frame, text="Recording Time (s):", font=("Arial", 9),
                bg=BG_COLOR, fg=PRIMARY_COLOR).pack()
        self.record_window = tk.DoubleVar(value=2.0)
        tk.Scale(recording_frame, from_=0.5, to=5.0, resolution=0.1, orient=tk.HORIZONTAL,
                variable=self.record_window, bg=SECONDARY_COLOR, fg=PRIMARY_COLOR,
                highlightbackground=BG_COLOR).pack(fill='x', pady=2)
        
        tk.Button(recording_frame, text="Record Gesture", bg=PRIMARY_COLOR, fg="white",
                 command=self.record_gesture, font=("Arial", 9, "bold")).pack(pady=2)
        tk.Button(recording_frame, text="Save to CSV", bg=CONFIDENCE_HIGH, fg="white",
                 command=self.save_gesture, font=("Arial", 9)).pack(pady=2)
        tk.Button(recording_frame, text="Discard", bg=CONFIDENCE_LOW, fg="white",
                 command=self.discard_gesture, font=("Arial", 9)).pack(pady=2)
        
        # ===== RIGHT COLUMN - Visualization =====
        right_column = tk.Frame(main_container, bg=BG_COLOR)
        right_column.pack(side="left", fill="both", expand=True, padx=5)
        
        # Recognition Display
        recognition_frame = tk.LabelFrame(right_column, text="Current Recognition",
                                         bg=BG_COLOR, fg=PRIMARY_COLOR,
                                         font=("Arial", 11, "bold"), padx=8, pady=2)
        recognition_frame.pack(fill="x", pady=5)
        
        self.gesture_display = tk.Label(recognition_frame, text="--",
                                        font=("Arial", 60, "bold"),
                                        bg=BG_COLOR, fg=PRIMARY_COLOR)
        self.gesture_display.pack(pady=5)
        
        self.gesture_name_label = tk.Label(recognition_frame, text="No gesture detected",
                                           font=("Arial", 12),
                                           bg=BG_COLOR, fg=PRIMARY_COLOR)
        self.gesture_name_label.pack()
        
        # Confidence Meter
        confidence_frame = tk.LabelFrame(right_column, text="Confidence",
                                        bg=BG_COLOR, fg=PRIMARY_COLOR,
                                        font=("Arial", 11, "bold"), padx=8, pady=5)
        confidence_frame.pack(fill="x", pady=5)
        
        self.confidence_label = tk.Label(confidence_frame, text="0%",
                                         font=("Arial", 18, "bold"),
                                         bg=BG_COLOR, fg=PRIMARY_COLOR)
        self.confidence_label.pack()
        
        # Confidence progress bar
        self.confidence_bar = ttk.Progressbar(confidence_frame, length=350,
                                             mode='determinate', maximum=100)
        self.confidence_bar.pack(pady=5)
        
        # Style the progress bar
        style = ttk.Style()
        style.theme_use('default')
        style.configure("green.Horizontal.TProgressbar", background=CONFIDENCE_HIGH)
        
        # Top 3 Predictions
        predictions_frame = tk.LabelFrame(right_column, text="Top 3 Predictions",
                                         bg=BG_COLOR, fg=PRIMARY_COLOR,
                                         font=("Arial", 11, "bold"), padx=8, pady=5)
        predictions_frame.pack(fill="x", pady=5)
        
        self.pred_labels = []
        for i in range(3):
            frame = tk.Frame(predictions_frame, bg=BG_COLOR)
            frame.pack(fill="x", pady=2)
            
            gesture_lbl = tk.Label(frame, text=f"#{i+1}: --", font=("Arial", 10),
                                  bg=BG_COLOR, fg=PRIMARY_COLOR, width=12, anchor="w")
            gesture_lbl.pack(side="left", padx=5)
            
            conf_lbl = tk.Label(frame, text="0%", font=("Arial", 10),
                               bg=BG_COLOR, fg=PRIMARY_COLOR, width=6)
            conf_lbl.pack(side="left")
            
            bar = ttk.Progressbar(frame, length=150, mode='determinate', maximum=100)
            bar.pack(side="left", padx=5)
            
            self.pred_labels.append((gesture_lbl, conf_lbl, bar))
        
        # Real-time Sensor Values
        sensor_frame = tk.LabelFrame(right_column, text="Live Sensor Values",
                                     bg=BG_COLOR, fg=PRIMARY_COLOR,
                                     font=("Arial", 11, "bold"), padx=8, pady=5)
        sensor_frame.pack(fill="x", pady=5)
        
        self.sensor_labels = []
        sensor_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        
        for i, name in enumerate(sensor_names):
            frame = tk.Frame(sensor_frame, bg=BG_COLOR)
            frame.pack(fill="x", pady=2)
            
            name_lbl = tk.Label(frame, text=f"{name}:", font=("Arial", 9, "bold"),
                               bg=BG_COLOR, fg=PRIMARY_COLOR, width=7, anchor="w")
            name_lbl.pack(side="left", padx=5)
            
            value_lbl = tk.Label(frame, text="0", font=("Arial", 9),
                                bg=BG_COLOR, fg=PRIMARY_COLOR, width=4)
            value_lbl.pack(side="left")
            
            bar = ttk.Progressbar(frame, length=250, mode='determinate', maximum=100)
            bar.pack(side="left", padx=10)
            
            self.sensor_labels.append((value_lbl, bar))

    # ===== Connection Methods =====
    def read_sensor_packet(self):
        """Read one JSON sensor packet from serial"""
        if self.ser and self.ser.in_waiting:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                data = json.loads(line)

                if "sensors" in data and len(data["sensors"]) == 5:
                    return data["sensors"]

            except json.JSONDecodeError:
                pass  # Ignore malformed lines

        return None



    def connect_glove(self):
        port = self.port_entry.get()
        try:
            self.ser = serial.Serial(port, 115200, timeout=1)
            time.sleep(2)
            self.status_label.config(text="Status: Connected", fg=CONFIDENCE_HIGH)
            messagebox.showinfo("Success", f"Connected to glove on {port}")
        except Exception as e:
            self.status_label.config(text="Status: Connection Failed", fg=CONFIDENCE_LOW)
            messagebox.showerror("Error", f"Failed to connect: {str(e)}")

            '''
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
        self.ble_thread.start()'''

    
    def calibrate_sensors(self):
        """Calibration routine: user opens and closes hand"""
        if not self.ser:
            messagebox.showwarning("Warning", "Connect glove first!")
            return
        
        result = messagebox.askokcancel("Calibration", 
            "Calibration Process:\n\n"
            "1. Click OK\n"
            "2. OPEN your hand fully (3 seconds)\n"
            "3. CLOSE your hand into a fist (3 seconds)\n\n"
            "This will calibrate sensor ranges.")
        
        if not result:
            return
        
        messagebox.showinfo("Step 1", "OPEN your hand fully!\n\nPress OK to start recording...")
        
        open_values = []
        '''
        for _ in range(30):
            if self.ser and self.ser.in_waiting:
                values = self.latest_data
                #if len(values) == 3: uncomment for 3 flex sensor testing
                if len(values) == 5: # comment for 3 flex sensor testing
                    open_values.append([float(v) for v in values])
            time.sleep(0.1)'''

        for _ in range(30):
            sensor_values = self.read_sensor_packet()
            if sensor_values:
                open_values.append(sensor_values)
            time.sleep(0.1)


        
        messagebox.showinfo("Step 2", "Now CLOSE your hand into a fist!\n\nPress OK to continue...")
        
        closed_values = []
        '''
        for _ in range(30):
            if self.ser and self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8').strip()
                values = line.split(',')
                #if len(values) == 3: uncomment for 3 flex sensor testing
                if len(values) == 5: # comment for 3 flex sensor testing
                    closed_values.append([float(v) for v in values])
            time.sleep(0.1)'''
        
        for _ in range(30):
            sensor_values = self.read_sensor_packet()
            if sensor_values:
                open_values.append(sensor_values)
            time.sleep(0.1)
        
        if open_values and closed_values:
            open_avg = np.mean(open_values, axis=0)
            closed_avg = np.mean(closed_values, axis=0)
            
            self.sensor_min = np.minimum(open_avg, closed_avg).tolist()
            self.sensor_max = np.maximum(open_avg, closed_avg).tolist()
            
            messagebox.showinfo("Success", "Calibration complete!")
        else:
            messagebox.showerror("Error", "Calibration failed - no data received")
    
    def notification_handler(self, sender, data):
        """Handles incoming BLE data"""
        try:
            line = data.decode().strip()
            values = line.split(',')
            if len(values) == 5:
                self.latest_data = values
        except Exception as e:
            print("BLE data error:", e)


    # ===== ML Methods =====
    def load_model(self):
        """Load trained KNN model and scaler"""
        try:
            self.model = joblib.load("models/knn_model.joblib")
            self.scaler = joblib.load("models/scaler.joblib")
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def start_recognition(self):
        """Start continuous gesture recognition"""
        if not self.ser:
            messagebox.showwarning("Warning", "Connect glove first!")
            return
        if not self.model:
            messagebox.showwarning("Warning", "Load model first!")
            return
        
        self.is_recognizing = True
        recognition_thread = threading.Thread(target=self.recognition_loop, daemon=True)
        recognition_thread.start()
    
    def stop_recognition(self):
        """Stop gesture recognition"""
        self.is_recognizing = False
        self.gesture_display.config(text="--")
        self.gesture_name_label.config(text="Recognition stopped")
        self.confidence_label.config(text="0%")
        self.confidence_bar['value'] = 0
    
    '''
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
            
        time.sleep(0.1)  # 10Hz update rate'''

    def recognition_loop(self):
        while self.is_recognizing:
            sensor_values = self.read_sensor_packet()

            if sensor_values:
                calibrated = self.apply_calibration(sensor_values)
                self.update_sensor_display(calibrated)
                self.sensor_buffer.append(calibrated)

                if len(self.sensor_buffer) >= self.smoothing_window.get():
                    smoothed = np.mean(
                        list(self.sensor_buffer)[-self.smoothing_window.get():],
                        axis=0
                    )
                    self.predict_gesture(smoothed)

            time.sleep(0.05)


    
    def apply_calibration(self, values):
        """Apply hand size and min/max calibration"""
        calibrated = []
        for i, v in enumerate(values):
            # Normalize to 0-100 range based on calibration
            normalized = (v - self.sensor_min[i]) / (self.sensor_max[i] - self.sensor_min[i]) * 100
            normalized = np.clip(normalized, 0, 100)
            # Apply hand size factor
            calibrated.append(normalized * self.hand_size.get())
        return calibrated
    
    def predict_gesture(self, sensor_values):
        """Predict gesture from sensor values with confidence"""
        try:
            # Scale the input
            X = np.array(sensor_values).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Get prediction with probabilities
            prediction = self.model.predict(X_scaled)[0]
            
            # Get prediction probabilities (for KNN, use predict_proba)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)[0]
                confidence = np.max(probabilities) * 100
                
                # Get top 3 predictions
                top_indices = np.argsort(probabilities)[-3:][::-1]
                top_predictions = [(self.model.classes_[i], probabilities[i] * 100) 
                                  for i in top_indices]
            else:
                confidence = 100  # If no probability available
                top_predictions = [(prediction, 100), ("--", 0), ("--", 0)]
            
            # Update display if confidence exceeds threshold
            if confidence >= self.confidence_threshold.get():
                self.gesture_display.config(text=prediction)
                self.gesture_name_label.config(text=f"Detected: {prediction}")
                self.confidence_label.config(text=f"{confidence:.1f}%")
                self.confidence_bar['value'] = confidence
                
                # Update confidence color
                if confidence >= 80:
                    self.confidence_label.config(fg=CONFIDENCE_HIGH)
                elif confidence >= 60:
                    self.confidence_label.config(fg=CONFIDENCE_MED)
                else:
                    self.confidence_label.config(fg=CONFIDENCE_LOW)
            else:
                self.gesture_display.config(text="?")
                self.gesture_name_label.config(text="Low confidence - hold steady")
            
            # Update top 3 predictions
            for i, (pred_label, conf_label, bar) in enumerate(self.pred_labels):
                if i < len(top_predictions):
                    gesture, conf = top_predictions[i]
                    pred_label.config(text=f"#{i+1}: {gesture}")
                    conf_label.config(text=f"{conf:.1f}%")
                    bar['value'] = conf
        
        except Exception as e:
            print(f"Prediction error: {e}")
    
    def update_sensor_display(self, values):
        """Update real-time sensor value display"""
        for i, (value_lbl, bar) in enumerate(self.sensor_labels):
            if i < len(values):
                value_lbl.config(text=f"{values[i]:.0f}")
                bar['value'] = min(values[i], 100)
    
    # ===== Recording Methods =====
    def record_gesture(self):
        """Record gesture data for training"""
        if not self.ser:
            messagebox.showwarning("Warning", "Connect glove first!")
            return
        
        label = self.gesture_label_entry.get()
        if not label:
            messagebox.showwarning("Warning", "Enter a gesture label!")
            return
        
        duration = self.record_window.get()
        messagebox.showinfo("Recording", f"Recording '{label}' for {duration} seconds...\n\nPress OK to start!")
        
        self.recorded_data = []
        start_time = time.time()
        
        '''
        while time.time() - start_time < duration:
            if self.ser and self.ser.in_waiting:
                values = self.latest_data
                if len(values) == 5:
                    sensor_values = [float(v) for v in values]
                    calibrated = self.apply_calibration(sensor_values)
                    self.recorded_data.append(calibrated + [label])
            time.sleep(0.05)
            '''
        
        while time.time() - start_time < duration:
            sensor_values = self.read_sensor_packet()

            if sensor_values:
                calibrated = self.apply_calibration(sensor_values)
                self.recorded_data.append(calibrated + [label])

            time.sleep(0.05)


        
        messagebox.showinfo("Complete", f"Recorded {len(self.recorded_data)} samples for '{label}'")
    
    def save_gesture(self):
        """Save recorded gesture to CSV"""
        if not self.recorded_data:
            messagebox.showwarning("Warning", "No data to save! Record first.")
            return
        
        label = self.gesture_label_entry.get()
        filename = f"data/raw/gesture_{label}_{int(time.time())}.csv"
        
        try:
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["thumb", "index", "middle", "ring", "pinky", "label"])
                writer.writerows(self.recorded_data)
            messagebox.showinfo("Success", f"Saved to {filename}")
            self.recorded_data = []
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def discard_gesture(self):
        """Discard recorded data"""
        self.recorded_data = []
        messagebox.showinfo("Discarded", "Recorded data discarded")

if __name__ == "__main__":
    app = GloveUI()
    app.mainloop()
