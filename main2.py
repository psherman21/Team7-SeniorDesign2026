# Philip Sherman
# Team 7 - Senior Design
# 2/2/2026
# v0.2.7

import tkinter as tk
from tkinter import ttk, messagebox
import serial, threading, time, joblib, ctypes
import numpy as np
import sys, os
from collections import deque
from pathlib import Path
from PIL import Image, ImageTk

# ── Shared sensor config (project root) ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sensor_config as sc

# Color Presets
SECONDARY_COLOR = "#ADD8E6"
PRIMARY_COLOR = "#1E3A8A"
BG_COLOR = "#E0E7EF"
CONFIDENCE_HIGH = "#4CAF50"
CONFIDENCE_MED = "#FFC107"
CONFIDENCE_LOW = "#F44336"

# ML / Smoothing Constants
K_NEIGHBORS = 5
SMOOTHING_WINDOW = 3

# Demo Mode Constants
DEMO_HOLD_DURATION  = 1.5   # seconds to hold a letter before confirming (~24 frames at 16Hz)
DEMO_COOLDOWN       = 1.0   # seconds after confirm before next letter can register (handles double letters)
DEMO_BOLD_FLASH     = 0.5   # seconds the newly confirmed letter stays bold
DEMO_MIN_CONFIDENCE = 60.0  # minimum confidence % to start the hold timer

class GloveUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Team 7 Senior Design Glove Demo")
        self.geometry("925x700")
        self.configure(bg=BG_COLOR)

        # Serial/ML variables
        self.ser = None
        self.model = None
        self.scaler = None
        self.is_recognizing = False
        self.sensor_buffer = deque(maxlen=20)
        self.recorded_data = []
        self.confusion_matrix = None

        # Sample rate tracking
        self.packet_count = 0
        self.last_rate_update = time.time()
        self.current_sample_rate = 0
        self.last_timestamp = None

        # Gesture stability filtering
        self.prediction_history = deque(maxlen=5)
        self.stable_gesture = None
        self.stability_threshold = 3

        # Calibration values
        self.sensor_min = [0, 0, 0, 0, 0]
        self.sensor_max = [100, 100, 100, 100, 100]
        self.calibration_done = False

        # Dataset selection
        self.selected_dataset = tk.StringVar(value="raw")

        # Demo mode state
        self.demo_mode        = False
        self.demo_word        = ""          # word built so far
        self.demo_state       = "WATCHING"  # WATCHING | HOLDING | COOLDOWN
        self.demo_hold_start  = None        # time when current hold began
        self.demo_cooldown_start = None     # time when cooldown began
        self.demo_current_letter = None     # letter currently being held

        self.setup_ui()

    def setup_ui(self):
        ctypes.windll.shcore.SetProcessDpiAwareness(True)
        # ===== Title with Logo =====
        title_frame = tk.Frame(self, bg=BG_COLOR)
        title_frame.pack(pady=2, fill="x", padx=10)

        # Sampling rate indicator
        rate_frame = tk.Frame(title_frame, bg=BG_COLOR)
        rate_frame.pack(pady=2, fill="x", padx=10)

        # Pulsing dot (canvas for animation)
        self.pulse_canvas = tk.Canvas(rate_frame, width=20, height=20, 
                                     bg=BG_COLOR, highlightthickness=0)
        self.pulse_canvas.pack(side="left", padx=5)
        self.pulse_dot = self.pulse_canvas.create_oval(5, 5, 15, 15, 
                                                       fill="gray", outline="")
        
        # Sample rate text
        self.rate_label = tk.Label(rate_frame, text="-- Hz", 
                                   font=("Arial", 11, "bold"),
                                   bg=BG_COLOR, fg=PRIMARY_COLOR)
        self.rate_label.pack(side="left")
        
        # Logo on the right
        logo_image = Image.open("assets/pitt_logo.png")
        
        # Resize pic to fit w/ title
        logo_height = 70
        aspect_ratio = logo_image.width / logo_image.height
        logo_width = int(logo_height * aspect_ratio)
        logo_image = logo_image.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
        
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = tk.Label(title_frame, image=logo_photo, bg=BG_COLOR)
        logo_label.image = logo_photo  # Keep a reference!
        logo_label.pack(side="right", padx=10)

        # Title on the left
        title_label = tk.Label(title_frame, text="Team 7 Senior Design Project Demo",
                               font=("Arial", 24, "bold"),
                               bg=BG_COLOR, fg=PRIMARY_COLOR)
        title_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # ===== Main Container =====
        main_container = tk.Frame(self, bg=BG_COLOR)
        main_container.pack(fill="both", expand=True, padx=4, pady=1)
        
        # ===== LEFT COLUMN - Preferences =====
        left_column = tk.Frame(main_container, bg=BG_COLOR)
        left_column.pack(side="left", fill="y", padx=10)
        
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
        self.port_entry.insert(0, "COM6")
        self.port_entry.pack(pady=2)
        
        self.connect_button = tk.Button(connection_frame, text="Connect", 
                                       bg=PRIMARY_COLOR, fg="white",
                                       command=self.toggle_connection, 
                                       font=("Arial", 9, "bold"),
                                       width=12)  # Fixed width prevents resizing
        self.connect_button.pack(side="left", padx=2)

        tk.Button(connection_frame, text="Calibrate", bg=PRIMARY_COLOR, fg="white",
                 command=self.calibrate_sensors, font=("Arial", 9, "bold")).pack(side="left",padx=2)

        tk.Button(connection_frame, text="⚙", bg=BG_COLOR, fg=PRIMARY_COLOR,
                 command=self.open_sensor_settings, font=("Arial", 12, "bold"),
                 relief="flat", width=2).pack(side="left", padx=2)
        
        # Recognition Control
        recognition_frame = tk.LabelFrame(left_column, text="Recognition Control",
                                         bg=BG_COLOR, fg=PRIMARY_COLOR,
                                         font=("Arial", 11, "bold"), padx=8, pady=2)
        recognition_frame.pack(fill="x", pady=2)

        # Dataset picker
        tk.Label(recognition_frame, text="Training Dataset:", font=("Arial", 9, "bold"),
                bg=BG_COLOR, fg=PRIMARY_COLOR).pack(pady=(4, 0))

        dataset_row = tk.Frame(recognition_frame, bg=BG_COLOR)
        dataset_row.pack(pady=2)

        self.dataset_dropdown = ttk.Combobox(dataset_row, textvariable=self.selected_dataset,
                                            state="readonly", font=("Arial", 9), width=13)
        self.dataset_dropdown.pack(side="left", padx=(0, 3))
        self.refresh_datasets()  # populate on startup

        tk.Button(dataset_row, text="↻", bg=BG_COLOR, fg=PRIMARY_COLOR,
                command=self.refresh_datasets, font=("Arial", 13, "bold"),
                width=2, relief="flat").pack(side="left")
        
        tk.Button(recognition_frame, text="Load Model", bg=PRIMARY_COLOR, fg="white",
                 command=self.load_model, font=("Arial", 9, "bold")).pack(pady=2)
        tk.Button(recognition_frame, text="Pause Recognition", bg=CONFIDENCE_LOW, fg="white",
                 command=self.stop_recognition, font=("Arial", 9, "bold")).pack(pady=2)

        # Demo mode button — only enabled after model is loaded
        self.demo_button = tk.Button(recognition_frame, text="Demo Mode",
                                     bg="#7B2D8B", fg="white",
                                     command=self.toggle_demo_mode,
                                     font=("Arial", 9, "bold"),
                                     state="disabled")
        self.demo_button.pack(pady=(4, 2))
        
        # Confidence Threshold
        conf_frame = tk.LabelFrame(left_column, text="Confidence Threshold",
                                   bg=BG_COLOR, fg=PRIMARY_COLOR,
                                   font=("Arial", 11, "bold"), padx=8, pady=2)
        conf_frame.pack(fill="x", pady=2)

        self.confidence_threshold = tk.DoubleVar(value=60.0)
        conf_scale = tk.Scale(conf_frame, from_=0, to=100, resolution=5, orient=tk.HORIZONTAL,
                             variable=self.confidence_threshold, bg=SECONDARY_COLOR,
                             fg=PRIMARY_COLOR, highlightbackground=BG_COLOR, length=120)
        conf_scale.pack(fill='x', pady=2)
        
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
        self.record_window_entry = tk.Entry(recording_frame, font=("Arial", 9),
                                            width=8, justify="center")
        self.record_window_entry.insert(0, "10")
        self.record_window_entry.pack(pady=2)
        
        tk.Button(recording_frame, text="Record Gesture", bg=PRIMARY_COLOR, fg="white",
                 command=self.record_gesture, font=("Arial", 9, "bold")).pack(pady=2)
        tk.Button(recording_frame, text="Save to CSV", bg=CONFIDENCE_HIGH, fg="white",
                 command=self.save_gesture, font=("Arial", 9)).pack(pady=2)
        
        # ===== RIGHT COLUMN =====
        right_column = tk.Frame(main_container, bg=BG_COLOR)
        right_column.pack(side="left", fill="both", expand=False, padx=5)

        # ===== TOP SECTION: side-by-side predictions | main display =====
        top_section = tk.Frame(right_column, bg=BG_COLOR)
        top_section.pack(fill="x", pady=(2, 0))

        # --- LEFT PANEL: Top 5 Predictions ---
        predictions_frame = tk.LabelFrame(top_section, text="Top 5 Predictions",
                                          bg=BG_COLOR, fg=PRIMARY_COLOR,
                                          font=("Arial", 11, "bold"), padx=6, pady=4)
        predictions_frame.pack(side="left", fill="none", expand=False, padx=(0, 4), anchor="n")

        self.pred_labels = []
        for i in range(5):
            row = tk.Frame(predictions_frame, bg=BG_COLOR)
            row.pack(fill="x", pady=3)

            gesture_lbl = tk.Label(row, text=f"#{i+1}: --", font=("Arial", 10, "bold"),
                                   bg=BG_COLOR, fg=PRIMARY_COLOR, width=10, anchor="w")
            gesture_lbl.pack(side="left", padx=(0, 3))

            bar = ttk.Progressbar(row, length=140, mode='determinate', maximum=100)
            bar.pack(side="left", padx=(0, 4))

            conf_lbl = tk.Label(row, text="0%", font=("Arial", 10),
                                bg=BG_COLOR, fg=PRIMARY_COLOR, width=5, anchor="w")
            conf_lbl.pack(side="left")

            self.pred_labels.append((gesture_lbl, conf_lbl, bar))

        # --- RIGHT PANEL: Main detection display ---
        detection_frame = tk.LabelFrame(top_section, text="Current Recognition",
                                        bg=BG_COLOR, fg=PRIMARY_COLOR,
                                        font=("Arial", 11, "bold"), padx=6, pady=4)
        detection_frame.pack(side="left", fill="y", expand=False, padx=(4, 0))

        # Big letter display
        self.gesture_display = tk.Label(detection_frame, text="--",
                                        font=("Arial", 72, "bold"),
                                        bg=BG_COLOR, fg=PRIMARY_COLOR)
        self.gesture_display.pack(pady=(2, 0), anchor="center")

        self.gesture_name_label = tk.Label(detection_frame, text="No gesture detected",
                                           font=("Arial", 13),
                                           bg=BG_COLOR, fg=PRIMARY_COLOR)
        self.gesture_name_label.pack(pady=(0, 2))

        # Confidence score + bar underneath the letter
        self.confidence_label = tk.Label(detection_frame, text="0%",
                                         font=("Arial", 18, "bold"),
                                         bg=BG_COLOR, fg=PRIMARY_COLOR)
        self.confidence_label.pack()

        style = ttk.Style()
        style.theme_use('default')
        style.configure("green.Horizontal.TProgressbar", background=CONFIDENCE_HIGH)

        self.confidence_bar = ttk.Progressbar(detection_frame, length=180,
                                              mode='determinate', maximum=100)
        self.confidence_bar.pack(pady=(2, 6))

        # ===== DEMO MODE PANEL (hidden until activated) =====
        self.demo_frame = tk.LabelFrame(right_column, text="Demo Mode — Spell Your Name!",
                                        bg=BG_COLOR, fg="#7B2D8B",
                                        font=("Arial", 11, "bold"), padx=8, pady=6)
        # Not packed yet — shown only when demo is active

        # Instruction label
        tk.Label(self.demo_frame, text="Hold each letter for 1.5 seconds to confirm",
                 font=("Arial", 9), bg=BG_COLOR, fg=PRIMARY_COLOR).pack(anchor="w")

        # Word display using Text widget (supports per-character bold tag)
        word_display_frame = tk.Frame(self.demo_frame, bg="white",
                                      relief="sunken", bd=1)
        word_display_frame.pack(fill="x", pady=(4, 2))

        self.demo_word_display = tk.Text(word_display_frame, height=2,
                                         font=("Arial", 28, "bold"),
                                         bg="white", fg=PRIMARY_COLOR,
                                         state="disabled", wrap="word",
                                         relief="flat", cursor="arrow")
        self.demo_word_display.pack(fill="x", padx=4, pady=4)
        # Tag for the bold flash on the newest letter
        self.demo_word_display.tag_configure("new_letter",
                                              font=("Arial", 28, "bold"),
                                              foreground=CONFIDENCE_HIGH)
        self.demo_word_display.tag_configure("normal",
                                              font=("Arial", 28, "bold"),
                                              foreground=PRIMARY_COLOR)

        # Hold progress bar + current letter label
        hold_row = tk.Frame(self.demo_frame, bg=BG_COLOR)
        hold_row.pack(fill="x", pady=(4, 2))

        tk.Label(hold_row, text="Holding:", font=("Arial", 9, "bold"),
                 bg=BG_COLOR, fg=PRIMARY_COLOR).pack(side="left", padx=(0, 4))

        self.demo_hold_letter_label = tk.Label(hold_row, text="--",
                                                font=("Arial", 14, "bold"),
                                                bg=BG_COLOR, fg=PRIMARY_COLOR,
                                                width=3)
        self.demo_hold_letter_label.pack(side="left", padx=(0, 6))

        self.demo_hold_bar = ttk.Progressbar(hold_row, length=200,
                                              mode='determinate', maximum=100)
        self.demo_hold_bar.pack(side="left")

        self.demo_status_label = tk.Label(hold_row, text="",
                                           font=("Arial", 9), bg=BG_COLOR,
                                           fg=PRIMARY_COLOR)
        self.demo_status_label.pack(side="left", padx=(6, 0))

        # Backspace + Clear buttons
        btn_row = tk.Frame(self.demo_frame, bg=BG_COLOR)
        btn_row.pack(fill="x", pady=(4, 0))

        tk.Button(btn_row, text="⌫  Backspace", bg=PRIMARY_COLOR, fg="white",
                  command=self.demo_backspace,
                  font=("Arial", 9, "bold")).pack(side="left", padx=(0, 6))
        tk.Button(btn_row, text="✕  Clear Word", bg=CONFIDENCE_LOW, fg="white",
                  command=self.demo_clear,
                  font=("Arial", 9, "bold")).pack(side="left")

        # Real-time Sensor Values
        sensor_frame = tk.LabelFrame(right_column, text="Live Sensor Values",
                                     bg=BG_COLOR, fg=PRIMARY_COLOR,
                                     font=("Arial", 11, "bold"), padx=4, pady=4)
        sensor_frame.pack(fill="none", pady=4, anchor="w")
        self.sensor_frame_ref = sensor_frame  # ref used by demo panel

        # Three equal columns: Flex | Accelerometer | Gyroscope
        LABEL_FONT  = ("Arial", 10, "bold")
        VALUE_FONT  = ("Arial", 10, "bold")
        HEADER_FONT = ("Arial", 10, "bold")
        ROW_PAD = 3

        flex_col  = tk.Frame(sensor_frame, bg=BG_COLOR)
        accel_col = tk.Frame(sensor_frame, bg=BG_COLOR)
        gyro_col  = tk.Frame(sensor_frame, bg=BG_COLOR)

        for col in (flex_col, accel_col, gyro_col):
            col.pack(side="left", fill="none", expand=False, padx=(0, 12))

        # --- Column headers ---
        tk.Label(flex_col,  text="Flex Sensors",  font=HEADER_FONT,
                 bg=BG_COLOR, fg=PRIMARY_COLOR).pack(anchor="w", pady=(0, 4))
        tk.Label(accel_col, text="Accelerometer", font=HEADER_FONT,
                 bg=BG_COLOR, fg=PRIMARY_COLOR).pack(anchor="w", pady=(0, 4))
        tk.Label(gyro_col,  text="Gyroscope",     font=HEADER_FONT,
                 bg=BG_COLOR, fg=PRIMARY_COLOR).pack(anchor="w", pady=(0, 4))

        # --- Flex sensors ---
        self.sensor_labels = []
        sensor_names = ["Pinky", "Ring", "Middle", "Index", "Thumb"]
        for name in sensor_names:
            row = tk.Frame(flex_col, bg=BG_COLOR)
            row.pack(fill="x", pady=ROW_PAD)
            tk.Label(row, text=f"{name}:", font=LABEL_FONT,
                     bg=BG_COLOR, fg=PRIMARY_COLOR, width=7, anchor="w").pack(side="left")
            value_lbl = tk.Label(row, text="0.00 V", font=VALUE_FONT,
                                 bg=BG_COLOR, fg=PRIMARY_COLOR, width=7, anchor="e")
            value_lbl.pack(side="left")
            bar = ttk.Progressbar(row, length=100, mode='determinate', maximum=100)
            bar.pack(side="left", padx=(4, 0))
            self.sensor_labels.append((value_lbl, bar))

        # --- Accelerometer ---
        self.imu_labels = {}
        for axis in ["X", "Y", "Z"]:
            row = tk.Frame(accel_col, bg=BG_COLOR)
            row.pack(fill="x", pady=ROW_PAD)
            tk.Label(row, text=f"Accel {axis}:", font=LABEL_FONT,
                     bg=BG_COLOR, fg=PRIMARY_COLOR, width=8, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="--", font=VALUE_FONT,
                           bg=BG_COLOR, fg=PRIMARY_COLOR, width=9, anchor="e")
            lbl.pack(side="left")
            self.imu_labels[f"acc_{axis.lower()}"] = lbl

        # --- Gyroscope ---
        for axis in ["X", "Y", "Z"]:
            row = tk.Frame(gyro_col, bg=BG_COLOR)
            row.pack(fill="x", pady=ROW_PAD)
            tk.Label(row, text=f"Gyro {axis}:", font=LABEL_FONT,
                     bg=BG_COLOR, fg=PRIMARY_COLOR, width=8, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="--", font=VALUE_FONT,
                           bg=BG_COLOR, fg=PRIMARY_COLOR, width=9, anchor="e")
            lbl.pack(side="left")
            self.imu_labels[f"gyr_{axis.lower()}"] = lbl

    # ===== Connection Methods =====
    def read_sensor_packet(self):
        """Read one serial packet and return sensor values as a list of floats.

        Uses sensor_config.parse_packet() so the expected format always matches
        whatever is set in sensor_config.json.  Timestamp (if active) is stored
        in self.last_timestamp but not returned.  Returns None on failure.
        """
        if not self.ser:
            return None
        try:
            line = self.ser.readline().decode("utf-8").strip()
            parsed = sc.parse_packet(line)
            if parsed is None:
                return None
            # Store timestamp if present
            if "__timestamp__" in parsed:
                self.last_timestamp = parsed["__timestamp__"]
            # Return sensor values in field order
            sensor_fields = sc.get_active_sensor_fields()
            return [parsed[f] for f in sensor_fields if f in parsed]
        except Exception as e:
            print(f"Read error: {e}")
            return None

    def toggle_connection(self):
        """Toggle between connect and disconnect"""
        if self.ser and self.ser.is_open:
            # Currently connected - disconnect
            self.disconnect_glove()
        else:
            # Currently disconnected - connect
            self.connect_glove()

    def connect_glove(self):
        """Connect to the glove"""
        port = self.port_entry.get()
        try:
            self.ser = serial.Serial(port, 115200, timeout=1)
            time.sleep(2)
            
            # Update UI
            self.status_label.config(text="Status: Connected", fg=CONFIDENCE_HIGH)
            self.connect_button.config(text="Disconnect", bg=CONFIDENCE_LOW)
            
            # Start data display automatically
            self.is_recognizing = True
            recognition_thread = threading.Thread(target=self.recognition_loop, daemon=True)
            recognition_thread.start()
            
        except Exception as e:
            self.status_label.config(text="Status: Connection Failed", fg=CONFIDENCE_LOW)
            messagebox.showerror("Error", f"Failed to connect: {str(e)}")

    def disconnect_glove(self):
        """Disconnect from the glove"""
        # Stop recognition loop
        self.is_recognizing = False
        time.sleep(0.1)  # Give thread time to stop
        
        # Close serial port
        if self.ser and self.ser.is_open:
            self.ser.close()
        
        # Update UI
        self.status_label.config(text="Status: Disconnected", fg="red")
        self.connect_button.config(text="Connect", bg=PRIMARY_COLOR)
        self.rate_label.config(text="-- Hz")
        self.pulse_canvas.itemconfig(self.pulse_dot, fill="gray")
        
        # Reset sensor displays
        for value_lbl, bar in self.sensor_labels:
            value_lbl.config(text="0.00 V")
            bar['value'] = 0
        for lbl in self.imu_labels.values():
            lbl.config(text="--")
    
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

        for _ in range(30):
            sensor_values = self.read_sensor_packet()
            if sensor_values:
                open_values.append(sensor_values)
            time.sleep(0.1)
        
        messagebox.showinfo("Step 2", "Now CLOSE your hand into a fist!\n\nPress OK to continue...")
        
        closed_values = []
        
        for _ in range(30):
            sensor_values = self.read_sensor_packet()
            if sensor_values:
                closed_values.append(sensor_values)
            time.sleep(0.1)
        
        if open_values and closed_values:
            open_avg = np.mean(open_values, axis=0)
            closed_avg = np.mean(closed_values, axis=0)
            
            self.sensor_min = np.minimum(open_avg, closed_avg).tolist()
            self.sensor_max = np.maximum(open_avg, closed_avg).tolist()
            
            self.calibration_done = True
            messagebox.showinfo("Success", "Calibration complete!")
        else:
            messagebox.showerror("Error", "Calibration failed - no data received")

    # ===== ML Methods =====
    def load_model(self):
        """Load trained KNN model and scaler"""
        try:
            self.model = joblib.load("models/knn_model.joblib")
            self.scaler = joblib.load("models/scaler.joblib")
            display_name = self.selected_dataset.get()
            full_path    = getattr(self, "_dataset_paths", {}).get(display_name, display_name)
            self.demo_button.config(state="normal")  # unlock demo mode
            messagebox.showinfo("Success", f"Model loaded!\nDataset: {display_name}\n({full_path})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def refresh_datasets(self):
        """Scan data/ folder and populate the dataset dropdown.
        Displays just the folder name (e.g. 'raw') but stores the
        full path in self._dataset_paths for use when saving.
        """
        base = Path(__file__).resolve().parent.parent / "data"
        base.mkdir(parents=True, exist_ok=True)
        (base / "raw").mkdir(exist_ok=True)
        (base / "processed").mkdir(exist_ok=True)

        # Build display-name -> full-path mapping
        self._dataset_paths = {
            p.name: str(p)
            for p in sorted(base.iterdir())
            if p.is_dir()
        }

        display_names = list(self._dataset_paths.keys())
        self.dataset_dropdown["values"] = display_names

        # Keep current selection if still valid, otherwise default to raw
        current = self.selected_dataset.get()
        if current not in display_names:
            self.selected_dataset.set("raw")
    
    def start_recognition(self):
        """Start continuous gesture recognition"""
        if not self.ser:
            messagebox.showwarning("Warning", "Connect glove first!")
            return
        
        if self.is_recognizing:
            return
        
        self.is_recognizing = True
        threading.Thread(target=self.recognition_loop, daemon=True).start()
    
    def stop_recognition(self):
        """Stop gesture recognition"""
        self.is_recognizing = False
        self.gesture_display.config(text="--")
        self.gesture_name_label.config(text="Recognition stopped")
        self.confidence_label.config(text="0%")
        self.confidence_bar['value'] = 0
    
    def recognition_loop(self):
        while self.is_recognizing:
            sensor_values = self.read_sensor_packet()

            if sensor_values:
                # Count packets for sample rate calculation
                self.packet_count += 1
                
                # Pulse the dot (green when receiving data)
                self.pulse_canvas.itemconfig(self.pulse_dot, fill=CONFIDENCE_HIGH)
                self.after(50, lambda: self.pulse_canvas.itemconfig(self.pulse_dot, fill="lightgreen"))
                
                # Update sample rate every second
                elapsed = time.time() - self.last_rate_update
                if elapsed >= 1.0:
                    self.current_sample_rate = self.packet_count / elapsed
                    self.rate_label.config(text=f"{self.current_sample_rate:.1f} Hz")
                    self.packet_count = 0
                    self.last_rate_update = time.time()
                
                # Show the raw voltage values in the display
                self.update_sensor_display(sensor_values)
                
                # Apply calibration for prediction purposes
                calibrated = self.apply_calibration(sensor_values)
                self.sensor_buffer.append(calibrated)

                # Only predict if model is loaded
                if self.model and len(self.sensor_buffer) >= SMOOTHING_WINDOW:
                    smoothed = np.mean(
                        list(self.sensor_buffer)[-SMOOTHING_WINDOW:],
                        axis=0
                    )
                    self.predict_gesture(smoothed)

                # Tick demo state machine if active
                if self.demo_mode:
                    self.after(0, self.demo_tick)

            time.sleep(0.05)

    def get_stable_prediction(self, prediction):
        self.prediction_history.append(prediction)

        counts = {}
        for p in self.prediction_history:
            counts[p] = counts.get(p, 0) + 1

        most_common = max(counts, key=counts.get)

        if counts[most_common] >= self.stability_threshold:
            self.stable_gesture = most_common

        return self.stable_gesture

    def apply_calibration(self, values):
        """Apply min/max calibration to flex sensors only.
        IMU values pass through unchanged. Output length matches
        however many sensor fields are active in sensor_config.json.
        """
        sensor_fields = sc.get_active_sensor_fields()
        calibrated = []
        flex_idx = 0
        for i, field in enumerate(sensor_fields):
            if i >= len(values):
                break
            if field.startswith("flex_"):
                range_val = self.sensor_max[flex_idx] - self.sensor_min[flex_idx]
                if range_val == 0:
                    normalized = 0.0
                else:
                    normalized = (values[i] - self.sensor_min[flex_idx]) / range_val * 100
                normalized = float(np.clip(normalized, 0, 100))
                calibrated.append(normalized)
                flex_idx += 1
            else:
                calibrated.append(values[i])
        return calibrated
    
    def predict_gesture(self, sensor_values):
        """Predict gesture from sensor values with confidence.
        Trims input to match however many features the scaler was trained on.
        """
        try:
            # Trim to scaler's expected feature count (handles flex-only vs flex+IMU)
            n = self.scaler.n_features_in_
            sensor_values = list(sensor_values)[:n]

            X = np.array(sensor_values).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Get prediction with probabilities
            prediction = self.model.predict(X_scaled)[0]

            stable_prediction = self.get_stable_prediction(prediction)

            if stable_prediction is None:
                return
            
            # Get prediction probabilities (use predict_proba for KNN)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)[0]
                confidence = np.max(probabilities) * 100
                
                # Get top 5 predictions
                top_indices = np.argsort(probabilities)[-5:][::-1]
                top_predictions = [(self.model.classes_[i], probabilities[i] * 100) 
                                  for i in top_indices]
            else:
                confidence = 100  # If no probability available
                top_predictions = [(prediction, 100)] + [("--", 0)] * 4
            
            # Update display if confidence exceeds threshold
            if confidence >= self.confidence_threshold.get():
                self.gesture_display.config(text=stable_prediction)
                self.gesture_name_label.config(text=f"Detected: {stable_prediction}")
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
        """Update real-time sensor value display.
        values[0:5]  = flex sensors (V)
        values[5:8]  = accelerometer (raw counts)
        values[8:11] = gyroscope (raw counts)
        """
        # Flex sensors
        for i, (value_lbl, bar) in enumerate(self.sensor_labels):
            if i < len(values):
                value_lbl.config(text=f"{values[i]:.2f} V")
                bar_value = max(0, min(100, (values[i] / 3.3) * 100))
                bar['value'] = bar_value

        # IMU readings
        if len(values) >= 11:
            axes = ["x", "y", "z"]
            for j, axis in enumerate(axes):
                accel_val = values[5 + j]
                gyro_val  = values[8 + j]
                self.imu_labels[f"acc_{axis}"].config(text=f"{accel_val:>8.1f}")
                self.imu_labels[f"gyr_{axis}"].config(text=f"{gyro_val:>8.1f}")
    
    # ===== Demo Mode Methods =====

    def toggle_demo_mode(self):
        """Enter or exit demo mode."""
        if self.demo_mode:
            self.exit_demo_mode()
        else:
            self.enter_demo_mode()

    def enter_demo_mode(self):
        """Activate demo mode: show panel, reset state."""
        if not self.model:
            messagebox.showwarning("Warning", "Load a model first!")
            return
        if not self.ser:
            messagebox.showwarning("Warning", "Connect the glove first!")
            return

        self.demo_mode = True
        self.demo_word = ""
        self.demo_state = "WATCHING"
        self.demo_hold_start = None
        self.demo_cooldown_start = None
        self.demo_current_letter = None

        # Show panel above sensor frame
        self.demo_frame.pack(fill="x", pady=(0, 4), before=self.sensor_frame_ref)

        # Update button label
        self.demo_button.config(text="■ Exit Demo", bg=CONFIDENCE_LOW)

        # Clear word display
        self._demo_set_word("", flash_last=False)
        self.demo_hold_bar['value'] = 0
        self.demo_hold_letter_label.config(text="--")
        self.demo_status_label.config(text="Sign a letter!")

        # Make sure recognition is running
        if not self.is_recognizing:
            self.start_recognition()

    def exit_demo_mode(self):
        """Deactivate demo mode: hide panel, reset button."""
        self.demo_mode = False
        self.demo_word = ""
        self.demo_frame.pack_forget()
        self.demo_button.config(text="▶ Demo Mode", bg="#7B2D8B")
        self.demo_hold_bar['value'] = 0
        self.demo_hold_letter_label.config(text="--")
        self.demo_status_label.config(text="")

    def demo_tick(self):
        """
        State machine tick — called from the recognition loop via self.after().
        Reads self.stable_gesture and confidence from the current prediction state.
        """
        if not self.demo_mode:
            return

        # Read current stable prediction and confidence from the main display
        letter      = self.stable_gesture
        conf_text   = self.confidence_label.cget("text")   # e.g. "82.3%"
        try:
            confidence = float(conf_text.replace("%", ""))
        except ValueError:
            confidence = 0.0

        now = time.time()
        valid = (letter is not None and
                 letter not in ("--", "?") and
                 confidence >= DEMO_MIN_CONFIDENCE)

        # ── COOLDOWN ──────────────────────────────────────────────────────────
        if self.demo_state == "COOLDOWN":
            elapsed = now - self.demo_cooldown_start
            remaining = max(0.0, DEMO_COOLDOWN - elapsed)
            self.demo_status_label.config(
                text=f"Next letter in {remaining:.1f}s...")
            self.demo_hold_bar['value'] = 0
            self.demo_hold_letter_label.config(text="--")
            if elapsed >= DEMO_COOLDOWN:
                self.demo_state = "WATCHING"
                self.demo_status_label.config(text="Sign a letter!")
            return

        # ── WATCHING ──────────────────────────────────────────────────────────
        if self.demo_state == "WATCHING":
            self.demo_hold_bar['value'] = 0
            if valid:
                # Start hold timer
                self.demo_state = "HOLDING"
                self.demo_hold_start = now
                self.demo_current_letter = letter
                self.demo_hold_letter_label.config(text=letter)
                self.demo_status_label.config(text="Hold steady...")
            else:
                self.demo_hold_letter_label.config(text="--")
                self.demo_status_label.config(text="Sign a letter!")
            return

        # ── HOLDING ───────────────────────────────────────────────────────────
        if self.demo_state == "HOLDING":
            # If letter changed or confidence dropped, reset
            if not valid or letter != self.demo_current_letter:
                self.demo_state = "WATCHING"
                self.demo_hold_start = None
                self.demo_current_letter = None
                self.demo_hold_bar['value'] = 0
                self.demo_hold_letter_label.config(text="--")
                self.demo_status_label.config(text="Sign a letter!")
                return

            elapsed = now - self.demo_hold_start
            progress = min(100.0, (elapsed / DEMO_HOLD_DURATION) * 100)
            self.demo_hold_bar['value'] = progress
            self.demo_hold_letter_label.config(text=letter)
            self.demo_status_label.config(
                text=f"{DEMO_HOLD_DURATION - elapsed:.1f}s...")

            if elapsed >= DEMO_HOLD_DURATION:
                # ── CONFIRM ───────────────────────────────────────────────────
                self.demo_word += letter
                self._demo_set_word(self.demo_word, flash_last=True)
                self.demo_hold_bar['value'] = 100
                self.demo_state = "COOLDOWN"
                self.demo_cooldown_start = now
                # Remove bold flash after DEMO_BOLD_FLASH seconds
                self.after(int(DEMO_BOLD_FLASH * 1000), self._demo_unflash)

    def _demo_set_word(self, word, flash_last=False):
        """Update the word Text widget. Optionally flash the last letter bold/green."""
        self.demo_word_display.config(state="normal")
        self.demo_word_display.delete("1.0", "end")
        if word:
            if flash_last and len(word) > 0:
                # All but last letter in normal style
                self.demo_word_display.insert("end", word[:-1], "normal")
                # Last letter in highlight style
                self.demo_word_display.insert("end", word[-1], "new_letter")
            else:
                self.demo_word_display.insert("end", word, "normal")
        self.demo_word_display.config(state="disabled")

    def _demo_unflash(self):
        """Return last letter to normal colour after bold flash."""
        if self.demo_mode:
            self._demo_set_word(self.demo_word, flash_last=False)

    def demo_backspace(self):
        """Remove the last confirmed letter."""
        if self.demo_word:
            self.demo_word = self.demo_word[:-1]
            self._demo_set_word(self.demo_word, flash_last=False)

    def demo_clear(self):
        """Clear the entire word."""
        self.demo_word = ""
        self._demo_set_word("", flash_last=False)
        self.demo_state = "WATCHING"
        self.demo_hold_bar['value'] = 0
        self.demo_hold_letter_label.config(text="--")
        self.demo_status_label.config(text="Sign a letter!")

    # ===== Sensor Settings Window =====

    def open_sensor_settings(self):
        """Open the sensor packet format settings Toplevel window."""
        # Only allow one instance
        if hasattr(self, "_settings_win") and self._settings_win.winfo_exists():
            self._settings_win.lift()
            return

        win = tk.Toplevel(self)
        win.title("Sensor Packet Settings")
        win.configure(bg=BG_COLOR)
        win.resizable(False, False)
        self._settings_win = win

        # ── Header ────────────────────────────────────────────────────────────
        tk.Label(win, text="Sensor Packet Format",
                 font=("Arial", 13, "bold"), bg=BG_COLOR,
                 fg=PRIMARY_COLOR).pack(pady=(12, 2), padx=16)
        tk.Label(win,
                 text="Select which fields your firmware sends.\n"
                      "Changes are saved to sensor_config.json and apply\n"
                      "to both the UI and data_logger.py.",
                 font=("Arial", 9), bg=BG_COLOR, fg=PRIMARY_COLOR,
                 justify="left").pack(padx=16, anchor="w")

        # ── Packet diagram canvas ──────────────────────────────────────────────
        diag_frame = tk.LabelFrame(win, text="Packet Diagram",
                                   bg=BG_COLOR, fg=PRIMARY_COLOR,
                                   font=("Arial", 10, "bold"), padx=6, pady=6)
        diag_frame.pack(fill="x", padx=16, pady=(10, 4))

        self._diag_canvas = tk.Canvas(diag_frame, height=40,
                                      bg=BG_COLOR, highlightthickness=0)
        self._diag_canvas.pack(fill="x")

        # ── Checkboxes ────────────────────────────────────────────────────────
        checks_outer = tk.Frame(win, bg=BG_COLOR)
        checks_outer.pack(padx=16, pady=(4, 8), anchor="w")

        cfg = sc.load_config()
        self._field_vars = {}

        groups = [
            ("Metadata",     ["timestamp"]),
            ("Flex Sensors", ["flex_1", "flex_2", "flex_3", "flex_4", "flex_5"]),
            ("Accelerometer",["accel_x", "accel_y", "accel_z"]),
            ("Gyroscope",    ["gyro_x",  "gyro_y",  "gyro_z"]),
        ]

        group_colors = sc.GROUP_COLORS
        group_map    = {f: g for g, fields in
                        [("meta", ["timestamp"]),
                         ("flex", ["flex_1","flex_2","flex_3","flex_4","flex_5"]),
                         ("accel",["accel_x","accel_y","accel_z"]),
                         ("gyro", ["gyro_x","gyro_y","gyro_z"])]
                        for f in fields}

        for col_idx, (group_name, fields) in enumerate(groups):
            col = tk.Frame(checks_outer, bg=BG_COLOR)
            col.grid(row=0, column=col_idx, padx=(0, 18), sticky="nw")

            color = group_colors[group_map[fields[0]]]
            tk.Label(col, text=group_name, font=("Arial", 9, "bold"),
                     bg=BG_COLOR, fg=color).pack(anchor="w", pady=(0, 2))

            for field in fields:
                var = tk.BooleanVar(value=cfg.get(field, False))
                self._field_vars[field] = var
                label = sc.FIELD_META[field]["label"]
                cb = tk.Checkbutton(col, text=label, variable=var,
                                    bg=BG_COLOR, fg=PRIMARY_COLOR,
                                    font=("Arial", 9),
                                    activebackground=BG_COLOR,
                                    command=self._redraw_diagram)
                cb.pack(anchor="w")

        # ── Expected packet length label ───────────────────────────────────────
        self._pkt_len_label = tk.Label(win, text="",
                                        font=("Arial", 9, "italic"),
                                        bg=BG_COLOR, fg=PRIMARY_COLOR)
        self._pkt_len_label.pack(pady=(0, 4))

        # ── Save / Cancel buttons ─────────────────────────────────────────────
        btn_row = tk.Frame(win, bg=BG_COLOR)
        btn_row.pack(pady=(4, 12))

        tk.Button(btn_row, text="Save", bg=PRIMARY_COLOR, fg="white",
                  font=("Arial", 10, "bold"), width=10,
                  command=lambda: self._save_sensor_settings(win)).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", bg=BG_COLOR, fg=PRIMARY_COLOR,
                  font=("Arial", 10), width=10,
                  command=win.destroy).pack(side="left", padx=6)

        # Draw diagram once window is rendered (bind to resize too)
        self._diag_canvas.bind("<Configure>", lambda e: self._redraw_diagram())

    def _redraw_diagram(self):
        """Redraw the packet diagram canvas based on current checkbox states."""
        if not hasattr(self, "_diag_canvas") or not self._diag_canvas.winfo_exists():
            return

        canvas = self._diag_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        total_w = canvas.winfo_width() or 520

        # Collect active fields in canonical order
        active = [f for f in sc.FIELD_META if self._field_vars.get(f, tk.BooleanVar()).get()]

        if not active:
            canvas.create_text(total_w // 2, 27, text="No fields selected",
                                fill="gray", font=("Arial", 10, "italic"))
            self._pkt_len_label.config(text="Expected packet length: 0 values")
            return

        box_w   = min(72, max(36, (total_w - 20) // len(active)))
        box_h   = 36
        y_top   = 9
        x_start = (total_w - box_w * len(active)) // 2

        for i, field in enumerate(active):
            meta  = sc.FIELD_META[field]
            group = meta["group"]
            color = sc.GROUP_COLORS[group]
            x0    = x_start + i * box_w
            x1    = x0 + box_w - 2
            y0    = y_top
            y1    = y_top + box_h

            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="white", width=1)
            # Short label
            short = meta["label"].replace("Accel ", "A").replace("Gyro ", "G").replace("Flex ", "F")
            canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                                text=short, fill="white",
                                font=("Arial", 8, "bold"))

        self._pkt_len_label.config(
            text=f"Expected packet length: {len(active)} value{'s' if len(active) != 1 else ''}")

    def _save_sensor_settings(self, win):
        """Write checkbox state to sensor_config.json and close the window."""
        new_cfg = {field: var.get() for field, var in self._field_vars.items()}
        sc.save_config(new_cfg)
        win.destroy()
        messagebox.showinfo("Saved",
            "Sensor config saved.\n\n"
            "The UI will use the new format immediately.\n"
            "Restart data_logger.py to pick up the change there.")

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
        
        try:
            duration = float(self.record_window_entry.get())
            if duration <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Warning", "Enter a valid recording duration (seconds).")
            return
        messagebox.showinfo("Recording", f"Recording '{label}' for {duration} seconds...\n\nPress OK to start!")
        
        self.recorded_data = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            sensor_values = self.read_sensor_packet()

            if sensor_values:
                calibrated = self.apply_calibration(sensor_values)
                # Store all 11 values (flex + IMU) for CSV
                self.recorded_data.append(calibrated + [label])

            time.sleep(0.05)
 
        messagebox.showinfo("Complete", f"Recorded {len(self.recorded_data)} samples for '{label}'")
    
    def save_gesture(self):
        """Save recorded gesture to CSV.
        Appends to an existing gesture_LABEL.csv in the selected dataset folder,
        or creates it with a header if it doesn't exist yet.
        """
        if not self.recorded_data:
            messagebox.showwarning("Warning", "No data to save! Record first.")
            return

        if not self.calibration_done:
            if not messagebox.askyesno("Warning",
                    "Calibration has not been run this session.\n"
                    "Data will be saved as raw voltages, not normalized 0-100.\n\n"
                    "Save anyway?"):
                return

        import csv

        label        = self.gesture_label_entry.get().upper()

        # Look up full path from display name
        display_name   = self.selected_dataset.get()
        full_path      = self._dataset_paths.get(display_name)
        if not full_path:
            messagebox.showerror("Error", f"Dataset '{display_name}' not found. Try refreshing.")
            return
        dataset_folder = Path(full_path)
        dataset_folder.mkdir(parents=True, exist_ok=True)

        filename   = dataset_folder / f"gesture_{label}.csv"
        file_exists = filename.exists()

        # Header matches active sensor fields from config
        sensor_fields = sc.get_active_sensor_fields()
        HEADER = sensor_fields + ["label"]

        print(f"[save_gesture] saving to: {filename}")

        try:
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(HEADER)
                writer.writerows(self.recorded_data)

            new_rows = len(self.recorded_data)
            messagebox.showinfo("Saved",
                f"Saved {new_rows} samples for '{label}'\n\nFile: {filename}")
            self.recorded_data = []
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def discard_gesture(self):
        self.recorded_data = []

if __name__ == "__main__":
    app = GloveUI()
    app.mainloop()