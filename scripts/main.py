#!/usr/bin/env python3
import base64
import json
import sys
import time
import requests
from PIL import Image, ImageTk
import cv2
import numpy as np
import logging
import tkinter as tk
from tkinter import ttk
import argparse
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Sensor Viewer')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for debug logs')
    parser.add_argument('--ip', type=str, default='10.10.10.1', help='Device IP address')
    return parser.parse_args()

class SensorViewer:
    def __init__(self, ip="10.10.10.1", debug=False, log_dir='logs'):
        # Setup logging
        if debug:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'sensor_viewer_{datetime.now():%Y%m%d_%H%M%S}.log')
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(level=logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
        self.ip = ip
        self.running = True
        self.current_topic = "camera"  # Default topic
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Sensor Viewer")
        self.setup_gui()
        
    def setup_gui(self):
        self.root.geometry("800x600")
        
        # Main frame
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(2, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        
        # Topic selector and status bar
        control_frame = ttk.Frame(self.frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Topic selector
        ttk.Label(control_frame, text="Topic:").pack(side=tk.LEFT, padx=5)
        self.topic_var = tk.StringVar(value="camera")
        topic_selector = ttk.Combobox(control_frame, 
                                    textvariable=self.topic_var,
                                    values=["camera", "tof"],
                                    state="readonly",
                                    width=10)
        topic_selector.pack(side=tk.LEFT, padx=5)
        topic_selector.bind('<<ComboboxSelected>>', self.on_topic_changed)
        
        # Status bar
        self.status_var = tk.StringVar(value="Status: Starting...")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Separator
        ttk.Separator(self.frame, orient='horizontal').grid(row=1, column=0, 
                                                          sticky=(tk.W, tk.E), 
                                                          pady=5)
        
        # Display frame for camera
        self.camera_frame = ttk.Frame(self.frame, width=640, height=480)
        self.camera_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.camera_frame.grid_propagate(False)
        
        self.image_label = ttk.Label(self.camera_frame)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Canvas for TOF visualization
        self.tof_canvas = tk.Canvas(self.frame, width=640, height=480, 
                                  bg='black', highlightthickness=0)
        self.tof_canvas.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.tof_canvas.grid_remove()  # Hidden by default
        
        # Control buttons
        btn_frame = ttk.Frame(self.frame)
        btn_frame.grid(row=3, column=0, pady=5)
        ttk.Button(btn_frame, text="Quit", command=self.quit).pack(side=tk.RIGHT)
        
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def on_topic_changed(self, event):
        self.current_topic = self.topic_var.get()
        if self.current_topic == "camera":
            self.tof_canvas.grid_remove()
            self.camera_frame.grid()
        else:
            self.camera_frame.grid_remove()
            self.tof_canvas.grid()
        self.status_var.set(f"Status: Switched to {self.current_topic}")
    
    def get_camera_frame(self):
        try:
            response = requests.post(
                f'http://{self.ip}/jsonrpc',
                json={
                    'id': 1,
                    'jsonrpc': '2.0',
                    'method': 'get_image_from_camera',
                    'params': {
                        'width': 324,
                        'height': 324,
                        'format': 'RGB',
                        'filter': 'BILINEAR',
                        'rotation': 270,
                        'auto_white_balance': False
                    }
                },
                headers={'Content-Type': 'application/json'},
                timeout=5.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting frame: {e}")
            return None

    def process_camera_frame(self, result):
        if not result or 'error' in result or 'result' not in result:
            return None
                
        try:
            result_data = result['result']
            width = result_data['width']
            height = result_data['height']
            frame_base64 = result_data['base64_data']
            
            frame = base64.b64decode(frame_base64)
            np_data = np.frombuffer(frame, dtype=np.uint8)
            rgb_image = np_data.reshape((height, width, 3))
            return Image.fromarray(rgb_image)
        except Exception as e:
            self.logger.error(f"Error processing camera frame: {e}")
            return None

    def get_tof_grid(self):
        try:
            response = requests.post(
                f'http://{self.ip}/jsonrpc',
                json={
                    'id': 1,
                    'jsonrpc': '2.0',
                    'method': 'get_tof_grid',
                    'params': []
                },
                headers={'Content-Type': 'application/json'},
                timeout=5.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting TOF frame: {e}")
            return None

    def draw_tof_grid(self, result):
        if not result or 'error' in result or 'result' not in result:
            return
        
        try:
            result_data = result['result']
            distances = result_data['distances']  # Array of 64 distance values
            temperature = result_data['temperature']  # Sensor temperature
            
            # Clear previous drawing
            self.tof_canvas.delete("all")
            
            # Calculate cell size based on canvas size
            canvas_width = self.tof_canvas.winfo_width()
            canvas_height = self.tof_canvas.winfo_height()
            cell_width = canvas_width / 8
            cell_height = canvas_height / 8
            
            max_distance = 4000  # 400cm in mm
            
            for row in range(8):
                for col in range(8):
                    x1 = col * cell_width
                    y1 = row * cell_height
                    x2 = x1 + cell_width
                    y2 = y1 + cell_height
                    
                    # Get distance value
                    distance = distances[row * 8 + col]
                    
                    # Calculate color (black to bright green based on distance)
                    # Ensure ratio is between 0 and 1
                    ratio = max(0, min(1, 1 - (distance / max_distance)))
                    # Convert to integer 0-255 and ensure it's within bounds
                    green = max(0, min(255, int(255 * ratio)))
                    # Format as 6-digit hex with leading zeros
                    color = f'#{0:02x}{green:02x}{0:02x}'
                    
                    # Draw cell
                    self.tof_canvas.create_rectangle(x1, y1, x2, y2, 
                                                fill=color, outline='gray')
                    
                    # Draw distance text
                    text_color = 'white' if green < 128 else 'black'
                    self.tof_canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                            text=f"{distance}" if distance > 0 else "---",
                                            fill=text_color,
                                            font=('Courier', 10))
            
            # Update status with temperature
            self.status_var.set(f"Status: TOF Running (Temp: {temperature}°C)")
                    
        except Exception as e:
            self.logger.error(f"Error drawing TOF frame: {e}", exc_info=True)

    def update_display(self):
        if not self.running:
            return

        if self.current_topic == "camera":
            response = self.get_camera_frame()
            if response:
                image = self.process_camera_frame(response)
                if image:
                    display_size = (640, 480)
                    image = image.resize(display_size, Image.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    self.image_label.configure(image=photo)
                    self.image_label.image = photo
                    self.status_var.set("Status: Camera Running")
        else:  # TOF
            response = self.get_tof_grid()
            if response:
                self.draw_tof_grid(response)
                self.status_var.set("Status: TOF Running")
            
        # Schedule next update
        self.root.after(33, self.update_display)  # ~30 FPS
        
    def quit(self):
        self.running = False
        self.root.quit()
        
    def run(self):
        # Start display updates
        self.root.after(0, self.update_display)
        self.root.mainloop()

def main():
    args = parse_args()
    viewer = SensorViewer(ip=args.ip, debug=args.debug, log_dir=args.log_dir)
    viewer.run()

if __name__ == '__main__':
    main()