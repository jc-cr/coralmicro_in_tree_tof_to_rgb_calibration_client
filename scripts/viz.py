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
from enum import Enum
import math

import sys

class Topic(Enum):
    CAMERA = "camera"
    TOF = "tof"
    ALIGNED_DEPTH_FRAME = "aligned_depth_frame"

class CameraSubscriber:
    """Handles camera data acquisition"""
    def __init__(self, ip="10.10.10.1"):
        self.ip = ip
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_frame(self):
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
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting camera frame: {e}")
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

class TofSubscriber:
    """Handles TOF data acquisition and visualization"""
    def __init__(self, ip="10.10.10.1"):
        self.ip = ip
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_distance = 4000  # 400cm in mm
        self.grid_size = (8, 8)

    def create_tof_visualization(self, result, size=(640, 480)):
        """Creates a PIL Image visualization of TOF data"""
        if not result or 'error' in result or 'result' not in result:
            return None
        
        try:
            result_data = result['result']
            distances = result_data['distances']
            temperature = result_data['temperature']
            
            # Create a blank image with proper size
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            
            # Calculate cell sizes
            cell_width = size[0] // self.grid_size[0]
            cell_height = size[1] // self.grid_size[1]
            
            for row in range(self.grid_size[1]):
                for col in range(self.grid_size[0]):
                    # Calculate cell position
                    x1 = col * cell_width
                    y1 = row * cell_height
                    x2 = x1 + cell_width
                    y2 = y1 + cell_height
                    
                    # Get distance value and calculate color
                    idx = row * self.grid_size[0] + col
                    distance = distances[idx]
                    ratio = max(0, min(1, 1 - (distance / self.max_distance)))
                    green = max(0, min(255, int(255 * ratio)))
                    
                    # Draw cell rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, green, 0), -1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    
                    # Add distance text
                    text_color = (255, 255, 255) if green < 128 else (0, 0, 0)
                    cv2.putText(img, 
                              f"{distance}",
                              (x1 + 5, y1 + cell_height//2),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.4,
                              text_color,
                              1)
                    
                    # Add cell ID (bottom left corner)
                    cv2.putText(img,
                              f"#{idx}",
                              (x1 + 5, y2 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.3,
                              text_color,
                              1)
            
            return Image.fromarray(img), temperature
            
        except Exception as e:
            self.logger.error(f"Error creating TOF visualization: {e}", exc_info=True)
            return None, None

    def get_frame(self):
        """Gets TOF data and returns visualization"""
        response = self.get_grid()
        if response:
            return self.create_tof_visualization(response)
        return None, None

    def get_grid(self):
        """Gets raw TOF data"""
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
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting TOF grid: {e}")
            return None


class AlignedDepthPublisher:
    """Handles alignment and processing of camera and TOF data"""
    def __init__(self, camera_subscriber, tof_subscriber):
        self.camera_subscriber = camera_subscriber
        self.tof_subscriber = tof_subscriber
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Camera parameters
        self.cam_width = 324
        self.cam_height = 324
        self.cam_fov = 110  # degrees
        self.tof_fov = 65   # degrees
        self.focal_length = (self.cam_width/2) / math.tan(math.radians(self.cam_fov/2))
        
        # Physical setup
        self.sensor_offset = (-1.4, 15.47, -13.15)  # mm
        self.max_distance = 4000  # mm

    def process_camera_frame(self, frame_data):
        if not frame_data or 'error' in frame_data or 'result' not in frame_data:
            return None
        
        try:
            result_data = frame_data['result']
            frame_base64 = result_data['base64_data']
            frame = base64.b64decode(frame_base64)
            np_data = np.frombuffer(frame, dtype=np.uint8)
            return np_data.reshape((result_data['height'], result_data['width'], 3))
        except Exception as e:
            self.logger.error(f"Error processing camera frame: {e}")
            return None

    def get_aligned_frame(self):
        camera_data = self.camera_subscriber.get_frame()
        tof_data = self.tof_subscriber.get_grid()
        
        if not camera_data or not tof_data:
            return None

        rgb_frame = self.process_camera_frame(camera_data)
        if rgb_frame is None:
            return None

        return self.align_depth_to_rgb(rgb_frame, tof_data)

    def align_depth_to_rgb(self, rgb_frame, tof_data):
        if not tof_data or 'error' in tof_data or 'result' not in tof_data:
            return None
            
        try:
            frame = rgb_frame.copy()
            overlay = frame.copy()
            debug_overlay = np.zeros_like(frame)  # For debugging projection
            
            # Get distances
            distances = tof_data['result']['distances']
            cell_points = []  # Store projected points for cell rendering
            
            # Project each TOF point
            for row in range(8):
                for col in range(8):
                    idx = row * 8 + col
                    depth = distances[idx]
                    
                    if depth <= 0:  # Skip invalid measurements
                        continue
                    
                    # Calculate ray angles from center
                    theta_x = math.radians((col - 3.5) * (self.tof_fov/8))
                    theta_y = math.radians((row - 3.5) * (self.tof_fov/8))
                    
                    # Create normalized ray vector
                    ray_x = math.sin(theta_x)
                    ray_y = math.sin(theta_y)
                    ray_z = math.sqrt(1 - ray_x*ray_x - ray_y*ray_y)
                    
                    # Get 3D point in ToF frame
                    x = depth * ray_x
                    y = depth * ray_y
                    z = depth * ray_z
                    
                    # Transform to camera frame
                    x_cam = x + self.sensor_offset[0]
                    y_cam = y + self.sensor_offset[1]
                    z_cam = z + self.sensor_offset[2]
                    
                    # Project to image plane
                    if z_cam > 0:
                        u = int(self.focal_length * (x_cam / z_cam) + self.cam_width/2)
                        v = int(self.focal_length * (y_cam / z_cam) + self.cam_height/2)
                        
                        # Store point if within frame
                        if 0 <= u < self.cam_width and 0 <= v < self.cam_height:
                            cell_points.append((u, v, depth, idx))
                            
                            # Draw debug point
                            cv2.circle(debug_overlay, (u, v), 2, (0, 255, 0), -1)
            
            # Draw cells and depth information
            for u, v, depth, idx in cell_points:
                # Calculate color based on depth
                ratio = max(0, min(1, 1 - (depth / self.max_distance)))
                green = max(0, min(255, int(255 * ratio)))
                color = (0, green, 0)  # BGR format
                
                # Draw cell with adaptive size based on depth
                cell_size = int(30 * (1000 / max(depth, 1000)))  # Larger cells for closer objects
                x1 = max(0, u - cell_size//2)
                y1 = max(0, v - cell_size//2)
                x2 = min(self.cam_width-1, u + cell_size//2)
                y2 = min(self.cam_height-1, v + cell_size//2)
                
                # Draw semi-transparent cell
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                # Add text info if cell is big enough
                if cell_size > 20:
                    # Distance text
                    cv2.putText(overlay, 
                              f"{depth}mm", 
                              (x1 + 2, y1 + cell_size//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.3,
                              (255, 255, 255),
                              1)
                    # Cell ID
                    cv2.putText(overlay, 
                              f"#{idx}", 
                              (x1 + 2, y2 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.3,
                              (255, 255, 255),
                              1)
            
            # Blend overlay with original frame
            alpha = 0.3
            result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Add debug visualization
            result = cv2.addWeighted(result, 0.7, debug_overlay, 0.3, 0)
            
            # Convert to PIL Image
            return Image.fromarray(result)
            
        except Exception as e:
            self.logger.error(f"Error aligning depth to RGB: {e}", exc_info=True)
            return None



class Visualizer:
    """Handles all visualization components"""
    def __init__(self, camera_subscriber, tof_subscriber, aligned_depth_publisher):
        self.camera_subscriber = camera_subscriber
        self.tof_subscriber = tof_subscriber
        self.aligned_publisher = aligned_depth_publisher
        self.current_topic = Topic.CAMERA.value
        self.current_frame = None
        
        self.root = tk.Tk()
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
        self.topic_var = tk.StringVar(value=Topic.CAMERA.value)
        topic_selector = ttk.Combobox(control_frame, 
                                    textvariable=self.topic_var,
                                    values=[t.value for t in Topic],
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
        
        # Display frame for all visualizations
        self.display_frame = ttk.Frame(self.frame, width=640, height=480)
        self.display_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.display_frame.grid_propagate(False)
        
        self.image_label = ttk.Label(self.display_frame)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Control buttons
        btn_frame = ttk.Frame(self.frame)
        btn_frame.grid(row=3, column=0, pady=5)

        ttk.Button(btn_frame, text="Quit", command=self.quit).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="Save Image", command=self.save_image).pack(side=tk.RIGHT)
        
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def on_topic_changed(self, event):
        self.current_topic = self.topic_var.get()
        self.status_var.set(f"Status: Switched to {self.current_topic}")
    
    def update_display(self):
        if self.current_topic == Topic.CAMERA.value:
            response = self.camera_subscriber.get_frame()
            if response:
                image = self.camera_subscriber.process_camera_frame(response)
                if image:
                    self.current_frame = image
                    display_size = (640, 480)
                    image = image.resize(display_size, Image.BILINEAR)
                    photo = ImageTk.PhotoImage(image)
                    self.image_label.configure(image=photo)
                    self.image_label.image = photo
                    self.status_var.set("Status: Camera Running")
                    
        elif self.current_topic == Topic.TOF.value:
            tof_image, temperature = self.tof_subscriber.get_frame()
            if tof_image:
                photo = ImageTk.PhotoImage(tof_image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                self.current_frame = tof_image
                self.status_var.set(f"Status: TOF Running (Temp: {temperature}°C)")

        elif self.current_topic == Topic.ALIGNED_DEPTH_FRAME.value:
            aligned_frame = self.aligned_publisher.get_aligned_frame()
            if aligned_frame:
                self.current_frame = aligned_frame
                display_size = (640, 480)
                aligned_frame = aligned_frame.resize(display_size, Image.LANCZOS)
                photo = ImageTk.PhotoImage(aligned_frame)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                self.status_var.set("Status: Aligned Depth Running")

        # Schedule next update
        self.root.after(33, self.update_display)  # ~30 FPS
        
    def save_image(self):
        img_filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        img_path = os.path.join("saved_images", img_filename)

        if self.current_frame:
            self.current_frame.save(img_path)
            self.status_var.set(f"Status: Image saved to {img_path}")

    def quit(self):
        self.root.quit()
        
    def run(self):
        # Start display updates
        self.root.after(0, self.update_display)
        self.root.mainloop()


def setup_logging(enable_logging):
    # Set base log level
    base_level = logging.DEBUG if enable_logging else logging.INFO
    
    # Configure root logger first
    root_logger = logging.getLogger()
    root_logger.setLevel(base_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    
    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(base_level)
    root_logger.addHandler(stream_handler)

    # File handler if logging is enabled
    if enable_logging:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"app_log_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

    # Log initial configuration
    logger = logging.getLogger(__name__)
    logger.debug("Logging configuration completed")
    logger.debug(f"Root logger level: {logging.getLevelName(root_logger.getEffectiveLevel())}")
    
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='Sensor Viewer')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--ip', type=str, default='10.10.10.1', help='Device IP address')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger = setup_logging(args.debug)
    
    try:
        logger.info("Starting application...")
        
        camera_subscriber = CameraSubscriber(ip=args.ip)
        tof_subscriber = TofSubscriber(ip=args.ip)
        aligned_depth_publisher = AlignedDepthPublisher(camera_subscriber, tof_subscriber)

        viewer = Visualizer(
            camera_subscriber=camera_subscriber,
            tof_subscriber=tof_subscriber, 
            aligned_depth_publisher=aligned_depth_publisher
        )
        
        viewer.run()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)