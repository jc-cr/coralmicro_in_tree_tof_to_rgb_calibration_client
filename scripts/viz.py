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


class CameraCalibrationParams:
    def __init__(self, calibration_json_path):
        try:
            self.calibration_json_path = calibration_json_path

            # Default calibration data
            self.camera_matrix = None
            self.distortion_coeffs = None
            self.image_size = None
            self.reprojection_error = None
            self.logger = logging.getLogger(self.__class__.__name__)

            self.__load_calibration_data()

        except Exception as e:
            self.logger.error(f"Error loading calibration data: {e}")
            raise e

    def __load_calibration_data(self):
        try:
            with open(self.calibration_json_path, 'r') as f:
                data = json.load(f)
                self.camera_matrix = np.array(data['camera_matrix'])
                self.distortion_coeffs = np.array(data['dist_coeffs'])
                self.image_size = tuple(data['image_size'])
                self.reprojection_error = data['reprojection_error']

            self.logger.info(f"Loaded calibration data from {self.calibration_json_path}")
            self.logger.info(f"Camera matrix: {self.camera_matrix}")
            self.logger.info(f"Distortion coefficients: {self.distortion_coeffs}")
            self.logger.info(f"Image size: {self.image_size}")
            self.logger.info(f"Reprojection error: {self.reprojection_error}")

        except Exception as e:
            raise e


class CameraSubscriber:
    """Handles camera data acquisition"""

    def __init__(self, ip="10.10.10.1", calibration_params=None, rotation=270, auto_white_balance=False):
        try:
            self.ip = ip
            self.logger = logging.getLogger(self.__class__.__name__)

            # User camera settings
            self.rotation = rotation
            self.auto_white_balance = auto_white_balance

            # Camera parameters
            self.camera_fov_diagonal = np.radians(110)
            self.camera_width = 324
            self.camera_height = 324
            self.camera_center_x = self.camera_width / 2
            self.camera_center_y = self.camera_height / 2

            # Camera calibration parameters
            self.calibration_params = calibration_params

        except Exception as e:
            self.logger.error(f"Error initializing camera subscriber: {e}")
            raise e

    def get_rgb_frame(self):
        result = self.__get_frame_data()

        if not result or 'error' in result or 'result' not in result:
            return None

        try:
            # Use data from the result to create a PIL Image
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

    def __get_frame_data(self):
        try:
            response = requests.post(
                f'http://{self.ip}/jsonrpc',
                json={
                    'id': 1,
                    'jsonrpc': '2.0',
                    'method': 'get_image_from_camera',
                    'params': {
                        'width': self.camera_width,
                        'height': self.camera_height,
                        'format': 'RGB',
                        'filter': 'BILINEAR',
                        'rotation': self.rotation,
                        'auto_white_balance': self.auto_white_balance,
                    }
                },
                headers={'Content-Type': 'application/json'},
                timeout=5.0
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting camera frame: {e}")
            return None


class TofSubscriber:
    """Handles TOF data acquisition and visualization"""

    def __init__(self, ip="10.10.10.1"):
        self.ip = ip
        self.logger = logging.getLogger(self.__class__.__name__)

        # TOF parameters
        self.tof_fov_horizontal = np.radians(45)
        self.tof_fov_vertical = np.radians(45)
        self.tof_grid_size = np.array([8, 8])

        self.max_distance = 4000  # 400cm in mm

    def get_tof_data(self):
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

    def get_tof_frame(self, size=(640, 480)):
        """Creates a PIL Image visualization of TOF data"""
        try:
            result = self.get_tof_data()

            if not result or 'error' in result or 'result' not in result:
                return None

            result_data = result['result']
            distances = result_data['distances']
            temperature = result_data['temperature']

            # Create a blank image with proper size
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

            # Calculate cell sizes
            cell_width = size[0] // self.tof_grid_size[0]
            cell_height = size[1] // self.tof_grid_size[1]

            for row in range(self.tof_grid_size[1]):
                for col in range(self.tof_grid_size[0]):
                    # Calculate cell position
                    x1 = col * cell_width
                    y1 = row * cell_height
                    x2 = x1 + cell_width
                    y2 = y1 + cell_height

                    # Get distance value and calculate color
                    idx = row * self.tof_grid_size[0] + col
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
            raise e


class AlignedDepthPublisher:
    """Handles alignment and processing of camera and TOF data"""

    def __init__(self, camera_subscriber, tof_subscriber):
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            self.camera_subscriber = camera_subscriber
            self.tof_subscriber = tof_subscriber

            # Physical setup
            # x, y, z offset of TOF sensor relative to camera
            # With board on ground, user looking at board from usbc side
            # x to the left
            # y to the top
            # z is up
            self.sensor_offset = np.array([1.4, 15.47, -13.15])
            self.ref_distance = 1000  # 1m in mm this comes from TOF sensor datasheet

            self.camera_matrix = camera_subscriber.calibration_params.camera_matrix
            # TOF parameters
            self.tof_grid_size = tof_subscriber.tof_grid_size
            self.tof_fov_h = tof_subscriber.tof_fov_horizontal
            self.tof_fov_v = tof_subscriber.tof_fov_vertical

            # Calculate and store all the fixed positions
            self.tof_corners = self.__calculate_tof_corners()
            self.camera_frame_corners = self.__transform_to_camera_frame(self.tof_corners)
            self.image_plane_corners = self.__project_to_image_plane(self.camera_frame_corners)
            self.grid_positions_2d = self.__create_grid_positions()
            self.cell_size = self.__calculate_cell_size()
            
            # Calculate and store cell pixel mappings
            self.cell_regions = self.__calculate_cell_pixels()
            
            # Generate header file if needed
            self.__generate_tof_rgb_mapping()


        except Exception as e:
            self.logger.error(f"Error initializing aligned depth publisher: {e}")
            raise e

    def __calculate_tof_corners(self):
        """Calculate TOF viewing frustum corners at 1m distance"""
        # At z = self.ref_distance (1000mm), calculate width and height using FOV
        width = 2 * self.ref_distance * np.tan(self.tof_fov_h / 2)
        height = 2 * self.ref_distance * np.tan(self.tof_fov_v / 2)

        # Define corners in TOF frame (bottom-left, bottom-right, top-left, top-right)
        corners = np.array([
            [-width/2, -height/2, self.ref_distance],  # bottom left
            [width/2, -height/2, self.ref_distance],   # bottom right
            [-width/2, height/2, self.ref_distance],   # top left
            [width/2, height/2, self.ref_distance]     # top right
        ])
        
        self.logger.debug(f"TOF corners at 1m: {corners}")
        return corners

    def __transform_to_camera_frame(self, points):
        """Transform points from TOF frame to camera frame"""
        # Add offset to each point
        camera_frame_points = points + self.sensor_offset
        
        self.logger.debug(f"Camera frame points: {camera_frame_points}")
        return camera_frame_points

    def __project_to_image_plane(self, points):
        """Project 3D points to camera image plane using pinhole model"""
        image_points = []
        
        for point in points:
            # Normalize by Z coordinate
            x_normalized = point[0] / point[2]
            y_normalized = point[1] / point[2]
            
            # Apply camera intrinsics
            u = self.camera_matrix[0, 0] * x_normalized + self.camera_matrix[0, 2]
            v = self.camera_matrix[1, 1] * y_normalized + self.camera_matrix[1, 2]
            
            image_points.append([int(u), int(v)])
        
        image_points = np.array(image_points)
        self.logger.debug(f"Image plane points: {image_points}")
        return image_points

    def __create_grid_positions(self):
        """Create grid positions from projected corners"""
        grid_positions = []
        corners = self.image_plane_corners
        
        # Extract corner coordinates
        bl, br, tl, tr = corners
        
        for i in range(self.tof_grid_size[0]):
            for j in range(self.tof_grid_size[1]):
                # Calculate normalized position within grid (0 to 1)
                x_ratio = i / (self.tof_grid_size[0] - 1)
                y_ratio = j / (self.tof_grid_size[1] - 1)
                
                # Interpolate position between corners
                left_point = bl + (tl - bl) * y_ratio
                right_point = br + (tr - br) * y_ratio
                point = left_point + (right_point - left_point) * x_ratio
                
                grid_positions.append({
                    'cell_id': i * self.tof_grid_size[0] + j,
                    'pos': (int(point[0]), int(point[1])),
                    'grid_pos': (i, j)
                })
        
        return grid_positions

    def __calculate_cell_size(self):
        """Calculate cell size for visualization based on projected corner distances"""
        corners = self.image_plane_corners
        # Calculate average width and height of cells based on corners
        width = np.linalg.norm(corners[1] - corners[0]) / self.tof_grid_size[0]
        height = np.linalg.norm(corners[2] - corners[0]) / self.tof_grid_size[1]
        return int((width + height) / 2)  # Average of width and height
    
    def __calculate_cell_pixels(self):
        """Calculate pixel regions for each TOF cell"""
        cell_regions = []
        
        for cell in self.grid_positions_2d:
            u, v = cell['pos']
            cell_id = cell['cell_id']
            i, j = cell['grid_pos']
            
            # Calculate pixel region for this cell
            half_size = self.cell_size // 2
            x_min = max(0, u - half_size)
            y_min = max(0, v - half_size)
            x_max = min(323, u + half_size)  # 324x324 image
            y_max = min(323, v + half_size)
            
            # Store both pixel indices and region bounds
            pixels = []
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    pixel_idx = y * 324 + x  # Convert 2D to 1D index
                    pixels.append(pixel_idx)
            
            cell_regions.append({
                'cell_id': cell_id,
                'grid_pos': (i, j),
                'pixels': pixels,
                'bounds': (x_min, y_min, x_max, y_max),
                'center': (u, v)
            })
        
        return cell_regions
    

    def __generate_tof_rgb_mapping(self):
        """Generate C++ header file with TOF cell to RGB pixel mappings"""
        header_content = []
        header_content.append("// Auto-generated TOF cell to RGB pixel mapping")
        header_content.append("// REV 0.0.1")
        header_content.append("#pragma once")
        header_content.append("#include <array>")
        header_content.append("#include <vector>")
        header_content.append("")
        header_content.append("namespace coralmicro {")
        header_content.append("")
        
        # Add constants
        header_content.append("// Number of TOF cells")
        header_content.append("constexpr size_t kTofCellCount = 64;  // 8x8 grid")
        header_content.append("")
        
        # Create the mapping structure with x,y coordinates
        header_content.append("struct PixelCoord {")
        header_content.append("    uint16_t x;")
        header_content.append("    uint16_t y;")
        header_content.append("};")
        header_content.append("")
        
        header_content.append("struct TofRgbMapping {")
        header_content.append("    std::vector<PixelCoord> pixels;")
        header_content.append("};")
        header_content.append("")
        
        # Create the lookup array
        header_content.append("// Mapping from TOF cell ID to corresponding RGB pixel coordinates")
        header_content.append("inline const std::array<TofRgbMapping, kTofCellCount> kTofRgbMap = {{")
        
        # Add each cell's mapping using stored cell_regions
        for region in sorted(self.cell_regions, key=lambda x: x['cell_id']):
            # Convert pixel indices to x,y coordinates
            coords = []
            for pixel_idx in region['pixels']:
                y = pixel_idx // 324  # Image width
                x = pixel_idx % 324
                coords.append(f"{{{{ {x}, {y} }}}}")
            
            coords_str = ", ".join(coords)
            header_content.append(f"    {{{{ {coords_str} }}}},  // Cell {region['cell_id']}")
        
        header_content.append("}};")
        header_content.append("")
        header_content.append("}  // namespace coralmicro")
        
        # Write to file
        with open("tof_rgb_mapping.hh", "w") as f:
            f.write("\n".join(header_content))
        
        return len(self.cell_regions)

    def get_aligned_frame(self):
        try:
            sensor_data = self.__get_sensor_data()
            if not sensor_data:
                return None

            rgb_frame, distances = sensor_data
            frame = np.array(rgb_frame)

            # Create visualization using pre-calculated grid positions
            return self.__create_visualization(frame, distances)

        except Exception as e:
            self.logger.error(f"Error getting aligned frame: {e}", exc_info=True)
            raise e

    def __get_sensor_data(self):
        tof_data = self.tof_subscriber.get_tof_data()
        rgb_frame = self.camera_subscriber.get_rgb_frame()

        if not rgb_frame or not tof_data:
            return None

        result_data = tof_data['result']
        distances = result_data['distances']

        return rgb_frame, distances

    def __create_visualization(self, frame, distances):
        try:
            overlay = frame.copy()
            distances_array = np.array(distances).reshape(self.tof_grid_size)
            
            # Use pre-calculated cell regions for visualization
            for region in self.cell_regions:
                i, j = region['grid_pos']
                x_min, y_min, x_max, y_max = region['bounds']
                u, v = region['center']
                cell_id = region['cell_id']
                
                if (0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]):
                    depth = distances_array[i, j]
                    
                    # Only draw if depth is valid
                    if 0 <= depth <= 4000:
                        intensity = int(255 * (1 - depth / 4000))
                        color = (0, intensity, 0)
                        
                        # Draw cell rectangle using stored bounds
                        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
                        
                        # Add cell ID
                        text_color = (255, 255, 255) if intensity > 128 else (0, 0, 0)
                        cv2.putText(overlay,
                                    f"#{cell_id}",
                                    (x_min + 2, y_min + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.3,
                                    text_color,
                                    1)
            
            result = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            return Image.fromarray(result)
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            return None


class Visualizer:
    """Handles all visualization components"""

    def __init__(self, camera_subscriber, tof_subscriber, aligned_depth_publisher):

        try:
            self.logger = logging.getLogger(self.__class__.__name__)

            # Subscribers and publishers
            self.camera_subscriber = camera_subscriber
            self.tof_subscriber = tof_subscriber
            self.aligned_publisher = aligned_depth_publisher

            self.current_topic = Topic.CAMERA.value

            self.current_frame = None

            # Display settings
            self.resize_flag = False
            self.resize_size = (640, 480)

            self.root = tk.Tk()
            self.setup_gui()

        except Exception as e:
            self.logger.error(f"Error initializing visualizer: {e}")
            raise e

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
        self.status_label = ttk.Label(
            control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # Separator
        ttk.Separator(self.frame, orient='horizontal').grid(row=1, column=0,
                                                            sticky=(
                                                                tk.W, tk.E),
                                                            pady=5)

        # Display frame for all visualizations
        self.display_frame = ttk.Frame(self.frame, width=640, height=480)
        self.display_frame.grid(row=2, column=0, padx=5,
                                pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.display_frame.grid_propagate(False)

        self.image_label = ttk.Label(self.display_frame)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Control buttons
        btn_frame = ttk.Frame(self.frame)
        btn_frame.grid(row=3, column=0, pady=5)

        ttk.Button(btn_frame, text="Quit",
                   command=self.quit).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="Save Image",
                   command=self.save_image).pack(side=tk.RIGHT)

        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def on_topic_changed(self, event):
        self.current_topic = self.topic_var.get()
        self.status_var.set(f"Status: Switched to {self.current_topic}")

    def update_display(self):
        if self.current_topic == Topic.CAMERA.value:
            image = self.camera_subscriber.get_rgb_frame()

            if image:
                # Store the current frame for saving
                self.current_frame = image

                if self.resize_flag:
                    # See https://zuru.tech/blog/the-dangers-behind-image-resizing for details on image resizing issues
                    image = image.resize(self.resize_size, Image.BILINEAR)

                photo = ImageTk.PhotoImage(image)

                self.image_label.configure(image=photo)
                self.image_label.image = photo
                self.status_var.set("Status: Camera Running")

        elif self.current_topic == Topic.TOF.value:
            result = self.tof_subscriber.get_tof_frame()

            if result:
                tof_image, temperature = result
                photo = ImageTk.PhotoImage(tof_image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                self.current_frame = tof_image
                self.status_var.set(
                    f"Status: TOF Running (Temp: {temperature}°C)")

        elif self.current_topic == Topic.ALIGNED_DEPTH_FRAME.value:
            aligned_frame = self.aligned_publisher.get_aligned_frame()

            if aligned_frame:
                # Store the current frame for saving
                self.current_frame = aligned_frame

                if self.resize_flag:
                    aligned_frame = aligned_frame.resize(
                        self.resize_size, Image.BILINEAR)

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
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument(
        '--ip', type=str, default='10.10.10.1', help='Device IP address')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger = setup_logging(args.debug)

    try:
        logger.info("Starting application...")

        camera_calibration_params = CameraCalibrationParams(
            "/app/camera_calibration.json")
        camera_subscriber = CameraSubscriber(
            ip=args.ip, calibration_params=camera_calibration_params)
        tof_subscriber = TofSubscriber(ip=args.ip)
        aligned_depth_publisher = AlignedDepthPublisher(
            camera_subscriber, tof_subscriber)

        viewer = Visualizer(
            camera_subscriber=camera_subscriber,
            tof_subscriber=tof_subscriber,
            aligned_depth_publisher=aligned_depth_publisher
        )

        viewer.run()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)
