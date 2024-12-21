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
    parser = argparse.ArgumentParser(description='Camera Viewer')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for debug logs')
    parser.add_argument('--ip', type=str, default='10.10.10.1', help='Device IP address')
    return parser.parse_args()

class CameraViewer:
    def __init__(self, ip="10.10.10.1", debug=False, log_dir='logs'):
        # Setup logging
        if debug:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'camera_viewer_{datetime.now():%Y%m%d_%H%M%S}.log')
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
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Camera Viewer")
        self.setup_gui()
        
    def setup_gui(self):
        self.root.geometry("800x600")
        
        # Main frame
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Status: Starting...")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Image display in a fixed-size frame
        self.display_frame = ttk.Frame(self.frame, width=640, height=480)
        self.display_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.display_frame.grid_propagate(False)
        
        self.image_label = ttk.Label(self.display_frame)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Control buttons
        btn_frame = ttk.Frame(self.frame)
        btn_frame.grid(row=2, column=0, pady=5)
        ttk.Button(btn_frame, text="Quit", command=self.quit).pack(side=tk.RIGHT)
        
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
    
    def get_camera_frame(self):
        try:
            response = requests.post(
                f'http://{self.ip}/jsonrpc',
                json={
                    'id': 1,
                    'jsonrpc': '2.0',
                    'method': 'get_synchronized_frame',
                    'params': []
                },
                headers={'Content-Type': 'application/json'},
                timeout=5.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting frame: {e}")
            return None
            
    def process_frame(self, result):
        if not result or 'error' in result or 'result' not in result:
            return None
            
        try:
            result_data = result['result']
            width = result_data['width']
            height = result_data['height']
            image_data = base64.b64decode(result_data['base64_data'])
            
            # Process image
            np_data = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width))
            rgb_image = cv2.cvtColor(np_data, cv2.COLOR_BAYER_RG2RGB)  # Direct conversion to RGB
            
            return Image.fromarray(rgb_image)
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return None
            
    def update_display(self):
        if not self.running:
            return
            
        response = self.get_camera_frame()
        if response:
            image = self.process_frame(response)
            if image:
                # Fixed display size
                display_size = (640, 480)
                image = image.resize(display_size, Image.LANCZOS)
                
                # Update display
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                self.status_var.set("Status: Running")
            
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
    viewer = CameraViewer(ip=args.ip, debug=args.debug, log_dir=args.log_dir)
    viewer.run()

if __name__ == '__main__':
    main()