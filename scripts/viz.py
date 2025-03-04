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

class TofMode(Enum):
        """
         mode: 
         #define VL53L8CX_RESOLUTION_4X4			((uint8_t) 16U 
         #define VL53L8CX_RESOLUTION_8X8			((uint8_t) 64U)
        """
        FOUR_X_FOUR = 16
        EIGHT_X_EIGHT = 64

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
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                result = self.__get_frame_data()
                if not result or 'error' in result or 'result' not in result:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None

                # Process frame data
                result_data = result['result']
                width = result_data['width']
                height = result_data['height']
                frame_base64 = result_data['base64_data']

                frame = base64.b64decode(frame_base64)
                np_data = np.frombuffer(frame, dtype=np.uint8)
                rgb_image = np_data.reshape((height, width, 3))

                return Image.fromarray(rgb_image)

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                self.logger.error(f"Error processing camera frame: {e}")
                return None

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

        # DEBUG file write flag
        self.tof_file_written_flag = False
        
        # Grid size will be determined dynamically
        self._grid_size = None
        self._mode = None  # Store current mode (4x4 or 8x8)
        self.min_distance = 0
        self.max_distance = 4000  # 400cm in mm

    def get_tof_data(self):
        """Gets raw TOF data from sensor with correct data types"""
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
            
            data = response.json()
            result = data['result']
            self._mode = result['mode']

            if self._mode == TofMode.FOUR_X_FOUR.value:
                self._grid_size = np.array([4, 4])
                array_size = 16
            elif self._mode == TofMode.EIGHT_X_EIGHT.value:
                self._grid_size = np.array([8, 8])
                array_size = 64
            else:
                self.logger.error(f"Invalid mode: {self._mode}")
                return None

            base64_str = result['results']
            raw_data = base64.b64decode(base64_str)

            # Read each field with its correct data type
            silicon_temp = np.frombuffer(raw_data, dtype=np.int8, offset=0, count=1)[0]  # int8_t
            ambient_per_spad = np.frombuffer(raw_data, dtype=np.uint32, offset=4, count=64)  # uint32_t[]
            nb_target_detected = np.frombuffer(raw_data, dtype=np.uint8, offset=260, count=64)  # uint8_t[]
            nb_spads_enabled = np.frombuffer(raw_data, dtype=np.uint32, offset=324, count=64)  # uint32_t[]
            signal_per_spad = np.frombuffer(raw_data, dtype=np.uint32, offset=580, count=64)  # uint32_t[]
            range_sigma_mm = np.frombuffer(raw_data, dtype=np.uint16, offset=836, count=64)  # uint16_t[]
            distances = np.frombuffer(raw_data, dtype=np.int16, offset=964, count=array_size)  # int16_t[]

            # For debugging, print the first few values of each field
            self.logger.debug(f"Temperature: {silicon_temp}°C")
            self.logger.debug(f"First 4 distances: {distances[:4]}")
            
            # Filter invalid measurements
            distances = np.where((distances < 0) | (distances > self.max_distance), 0, distances)

            return {
                'mode': self._mode,
                'distances': distances
            }

        except Exception as e: 
            self.logger.error(f"Error getting TOF data: {e}")
            return None

            
    def get_tof_frame(self, size=(640, 480)):
        """Creates a PIL Image visualization of TOF data"""
        try:
            result = self.get_tof_data()
            if result is None:
                return None

            # Process frame data
            distances = result['distances']

            # Create visualization
            return self.__create_visualization(distances, size)

        except Exception as e:
            self.logger.error(f"Error creating TOF visualization: {e}", exc_info=True)
            return None


    def get_depth_color(self, depth):
        """Convert depth to RGB color using a multi-color gradient with improved validation
        
        Args:
            depth: Depth value in mm
            max_depth: Maximum depth value in mm (default 4000)
        
        Returns:
            Tuple of (B, G, R) values for OpenCV or None if invalid
        """
        # More strict validation of depth values
        if depth < self.min_distance or depth > self.max_distance:
            norm_depth = 0

        else:
            # Normalize valid depth to 0-1
            norm_depth = depth / self.max_distance
        
        # Create a rainbow gradient
        if norm_depth < 0.25:  # Red (close) -> Yellow
            ratio = norm_depth * 4
            return (0, int(255 * ratio), 255)
        elif norm_depth < 0.5:  # Yellow -> Green
            ratio = (norm_depth - 0.25) * 4
            return (0, 255, int(255 * (1 - ratio)))
        elif norm_depth < 0.75:  # Green -> Cyan
            ratio = (norm_depth - 0.5) * 4
            return (int(255 * ratio), 255, 0)
        else:  # Cyan -> Blue
            ratio = (norm_depth - 0.75) * 4
            return (255, int(255 * (1 - ratio)), 0)


    def __create_visualization(self, distances, size):
        """Create TOF visualization with improved validation and error handling"""
        try:
            # Create a blank image
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

            if self._grid_size is None:
                self.logger.error("Grid size not set. Cannot create visualization.")
                return None

            # Calculate cell sizes
            display_aspect = size[0] / size[1]
            grid_aspect = self._grid_size[0] / self._grid_size[1]
            
            if display_aspect > grid_aspect:
                cell_height = size[1] // self._grid_size[1]
                cell_width = cell_height
                x_offset = (size[0] - (cell_width * self._grid_size[0])) // 2
                y_offset = 0
            else:
                cell_width = size[0] // self._grid_size[0]
                cell_height = cell_width
                x_offset = 0
                y_offset = (size[1] - (cell_height * self._grid_size[1])) // 2

            # Reshape distances with validation
            try:
                distances_array = np.array(distances).reshape(self._grid_size)
            except ValueError as e:
                self.logger.error(f"Error reshaping distances array: {e}")
                return None

            # Adjust font size based on mode
            base_font_scale = 0.3 if self._mode == TofMode.EIGHT_X_EIGHT.value else 0.5
            font_scale = base_font_scale * (cell_width / 100)

            # Draw grid cells with improved validation
            for row in range(self._grid_size[1]):
                for col in range(self._grid_size[0]):
                    x1 = x_offset + col * cell_width
                    y1 = y_offset + row * cell_height
                    x2 = x1 + cell_width
                    y2 = y1 + cell_height

                    cell_id = row * self._grid_size[0] + (self._grid_size[0] - 1 - col)
                    distance = distances_array[row, col]

                    # Draw cell outline regardless of validity
                    cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), 1)

                    # Get color for valid distances
                    color = self.get_depth_color(distance)
                    if color is not None:
                        # Valid distance - draw filled cell
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                        
                        # Calculate text color based on background brightness
                        brightness = (0.2126 * color[2] + 0.7152 * color[1] + 0.0722 * color[0]) / 255
                        text_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)
                        
                        # Add cell ID and valid distance
                        text_y_offset = int(cell_height * 0.2)
                        cv2.putText(img,
                                f"#{cell_id}",
                                (x1 + 2, y1 + text_y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                text_color,
                                1)
                        cv2.putText(img,
                                f"{distance}mm",
                                (x1 + 2, y1 + text_y_offset * 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                text_color,
                                1)
                    else:
                        # Invalid distance - just show cell ID in grey
                        cv2.putText(img,
                                f"#{cell_id}",
                                (x1 + 2, y1 + int(cell_height * 0.2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                (128, 128, 128),
                                1)

            return Image.fromarray(img)

        except Exception as e:
            self.logger.error(f"Error in create_visualization: {e}", exc_info=True)
            return None

class AlignedDepthPublisher:
    """Handles alignment and processing of camera and TOF data"""

    def __init__(self, camera_subscriber, tof_subscriber):
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            self.camera_subscriber = camera_subscriber
            self.tof_subscriber = tof_subscriber

            # Physical setup
            # See thesis for coordinate ref frame
            self.sensor_offset = np.array([-0.85, 27.33, -12.32])
            self.ref_distance = 1000  # 1m in mm

            self.camera_matrix = camera_subscriber.calibration_params.camera_matrix
            
            # Wait for first TOF data to determine grid size
            initial_data = self.tof_subscriber.get_tof_data()
            if not initial_data:
                raise Exception("Error getting initial TOF data")

            # TOF parameters
            self.tof_fov_h = tof_subscriber.tof_fov_horizontal
            self.tof_fov_v = tof_subscriber.tof_fov_vertical
            self.tof_grid_size = tof_subscriber._grid_size

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

    def __calculate_tof_corners(self):
        """Calculate TOF viewing frustum corners at 1m distance"""
        # At z = self.ref_distance (1000mm), calculate width and height using FOV
        width = 2 * self.ref_distance * np.tan(self.tof_fov_h / 2)
        height = 2 * self.ref_distance * np.tan(self.tof_fov_v / 2)

        self.logger.debug(f"TOF width: {width}, height: {height}")

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
        """Create grid positions from projected corners with correct TOF orientation"""
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
                
                # Calculate cell ID based on TOF orientation
                # Cell 0 should be top-right, incrementing left then down
                cell_id = j * self.tof_grid_size[0] + (self.tof_grid_size[0] - 1 - i)
                
                grid_positions.append({
                    'cell_id': cell_id,
                    'pos': (int(point[0]), int(point[1])),
                    'grid_pos': (i, j),  # Store original grid position for visualization
                    'tof_pos': (j, i)  # Store TOF position for mapping
                })
        
        return grid_positions

    def __calculate_cell_size(self):
        """Calculate cell size that ensures complete coverage"""
        corners = self.image_plane_corners
        # Calculate max width and height of cells to ensure coverage
        width = np.linalg.norm(corners[1] - corners[0]) / (self.tof_grid_size[0] - 1)  # Use n-1 for full coverage
        height = np.linalg.norm(corners[2] - corners[0]) / (self.tof_grid_size[1] - 1)
        # Use the larger size to ensure overlap rather than gaps
        return int(max(width, height))

    def __calculate_cell_pixels(self):
        """Calculate overlapping pixel regions for each TOF cell"""
        cell_regions = []
        
        for cell in self.grid_positions_2d:
            u, v = cell['pos']
            cell_id = cell['cell_id']
            i, j = cell['grid_pos']
            tof_row, tof_col = cell['tof_pos']
            
            # Calculate cell bounds with slight overlap
            cell_size = self.cell_size
            x_min = max(0, u - cell_size//2)
            y_min = max(0, v - cell_size//2)
            x_max = min(323, u + cell_size//2)
            y_max = min(323, v + cell_size//2)
            
            # Ensure minimum cell size
            if x_max - x_min < cell_size:
                if x_min == 0:
                    x_max = min(323, x_min + cell_size)
                else:
                    x_min = max(0, x_max - cell_size)
            
            if y_max - y_min < cell_size:
                if y_min == 0:
                    y_max = min(323, y_min + cell_size)
                else:
                    y_min = max(0, y_max - cell_size)
            
            # Store both pixel indices and region bounds
            pixels = []
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    pixel_idx = y * 324 + x
                    pixels.append(pixel_idx)
            
            cell_regions.append({
                'cell_id': cell_id,
                'grid_pos': (i, j),
                'tof_pos': (tof_row, tof_col),
                'pixels': pixels,
                'bounds': (x_min, y_min, x_max, y_max),
                'center': (u, v)
            })
        
        return cell_regions

    def __generate_tof_rgb_mapping(self):
        """Generate optimized C++ header file with TOF cell to RGB pixel mappings using bounding boxes"""
        header_content = []
        header_content.append("// Auto-generated TOF cell to RGB pixel mapping")
        header_content.append("#pragma once")
        header_content.append("#include <array>")
        header_content.append("#include <cstdint>")
        header_content.append("")
        header_content.append("namespace coralmicro {")
        header_content.append("")
        
        # Get total number of cells based on grid size
        total_cells = self.tof_grid_size[0] * self.tof_grid_size[1]
        mode_str = "4x4" if total_cells == 16 else "8x8"
        
        # Add constants
        header_content.append("// Number of TOF cells")
        header_content.append(f"constexpr size_t kTofCellCount = {total_cells};  // {mode_str} grid")
        header_content.append("")
        
        # Define compact cell region structure
        header_content.append("struct TofCellRegion {")
        header_content.append("    uint16_t x_min;")
        header_content.append("    uint16_t y_min;")
        header_content.append("    uint16_t x_max;")
        header_content.append("    uint16_t y_max;")
        header_content.append("    uint16_t center_x;")
        header_content.append("    uint16_t center_y;")
        header_content.append("    uint32_t area;      // Pre-calculated area of the cell region")
        header_content.append("};")
        header_content.append("")
        
        # Create array of cell regions
        header_content.append("// Mapping of TOF cell regions (bounds, center, and area)")
        header_content.append(f"constexpr std::array<TofCellRegion, kTofCellCount> kTofCellRegions = {{")
        
        for cell_idx in range(total_cells):
            region = next(r for r in self.cell_regions if r['cell_id'] == cell_idx)
            x_min, y_min, x_max, y_max = region['bounds']
            center_x, center_y = region['center']
            area = (x_max - x_min + 1) * (y_max - y_min + 1)  # Calculate cell area
            
            header_content.append(f"    {{ {x_min}, {y_min}, {x_max}, {y_max}, {center_x}, {center_y}, {area} }},  // Cell {cell_idx}")
        
        header_content.append("}};")
        header_content.append("")
        
        header_content.append("")
        
        header_content.append("}  // namespace coralmicro")
        
        # Write to file
        with open("tof_rgb_mapping.hh", "w") as f:
            f.write("\n".join(header_content))
        
        return total_cells
    
    def __back_project_to_camera_frame(self, image_points, z_distance=1000):
        """Back-project image points to camera frame at specified Z distance
        
        Args:
            image_points: Array of [u,v] image coordinates
            z_distance: Z distance in mm to project to (default 1m = 1000mm)
            
        Returns:
            Array of [x,y,z] camera frame coordinates in mm
        """
        # Get inverse of camera matrix
        K_inv = np.linalg.inv(self.camera_matrix)
        
        camera_points = []
        for point in image_points:
            # Convert to homogeneous coordinates
            uv_homog = np.array([point[0], point[1], 1])
            
            # Back project to normalized coordinates
            xyz_normalized = K_inv.dot(uv_homog)
            
            # Scale to desired Z distance
            scale = z_distance / xyz_normalized[2]
            xyz_camera = xyz_normalized * scale
            
            camera_points.append(xyz_camera)
        
        return np.array(camera_points)

    def __get_sensor_data(self):
        """Get synchronized sensor data from both TOF and RGB camera"""
        tof_data = self.tof_subscriber.get_tof_data()
        rgb_frame = self.camera_subscriber.get_rgb_frame()

        if not rgb_frame or not tof_data:
            return None

        return rgb_frame, tof_data['distances']

    def __create_visualization(self, frame, distances):
        """Create aligned depth visualization with improved color mapping"""
        try:
            overlay = frame.copy()
            grid_size = self.tof_grid_size
            
            # Validate grid size matches distances array
            expected_size = grid_size[0] * grid_size[1]
            if len(distances) != expected_size:
                self.logger.error(f"Distance array size mismatch. Expected {expected_size}, got {len(distances)}")
                return None

            # Reshape distances array based on grid size
            distances_array = np.array(distances).reshape(grid_size)
            
            # Adjust visualization parameters based on grid size
            font_scale = 0.3 if grid_size[0] == 8 else 0.5
            opacity = 0.5  # Blend factor for overlay
            
            # Use pre-calculated cell regions for visualization
            for region in self.cell_regions:
                i, j = region['grid_pos']
                tof_row, tof_col = region['tof_pos']
                x_min, y_min, x_max, y_max = region['bounds']
                u, v = region['center']
                cell_id = region['cell_id']
                
                # Ensure point is within frame bounds
                if (0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]):
                    depth = distances_array[tof_row, tof_col]
                    
                    # Only visualize valid depth measurements
                    if 0 < depth <= self.tof_subscriber.max_distance:
                        # Get color based on depth
                        color = self.tof_subscriber.get_depth_color(depth)
                        
                        # Draw cell overlay
                        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
                        
                        # Calculate text color based on background brightness
                        brightness = (0.2126 * color[2] + 0.7152 * color[1] + 0.0722 * color[0]) / 255
                        text_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)
                        
                        # Add cell ID and depth with mode-appropriate font size
                        cv2.putText(overlay,
                                f"#{cell_id}",
                                (x_min + 2, y_min + 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                text_color,
                                1)
                        cv2.putText(overlay,
                                f"{depth}mm",
                                (x_min + 2, y_min + 24),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                text_color,
                                1)
            
            # Blend overlay with original frame
            result = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
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
            result = self.tof_subscriber.get_tof_frame(size=(self.camera_subscriber.camera_width, self.camera_subscriber.camera_height))

            if result:
                tof_image = result
                photo = ImageTk.PhotoImage(tof_image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                self.current_frame = tof_image

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
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 

        img_filename = f"image_{current_time}.png"
        img_path = os.path.join("saved_images", img_filename)

        if self.current_frame:
            self.current_frame.save(img_path)

            # Save depth map if available as JSON for human readable
            if self.current_topic == Topic.TOF.value or self.current_topic == Topic.ALIGNED_DEPTH_FRAME.value:
                depth_filename = f"depth_{current_time}.json"
                depth_path = os.path.join("saved_images", depth_filename)

                with open(depth_path, 'w') as f:
                    np_data = np.array(self.tof_subscriber.get_tof_data()['distances'])
                    json.dump(np_data.tolist(), f)

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
