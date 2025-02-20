import numpy as np
import cv2 as cv
import glob
import json
import os

def calibrate_camera(image_folder, checkerboard_size=(7,6)):
    # Set termination criteria for cornerSubPix
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all images
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    
    # Get list of all calibration images
    images = glob.glob(f'{image_folder}/*.jpg') + glob.glob(f'{image_folder}/*.png')
    
    successful_images = 0
    image_size = None
    
    print(f"Found {len(images)} images. Processing...")
    print(f"Looking for a {checkerboard_size[0]}x{checkerboard_size[1]} internal corner pattern")
    
    for fname in images:
        print(f"\nProcessing image: {fname}")
        img = cv.imread(fname)
        if img is None:
            print(f"Could not read image {fname}")
            continue
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1]
            print(f"Image size: {image_size}")
        
        # Let's save the grayscale image for debugging
        debug_dir = 'cal_debug'
        os.makedirs(debug_dir, exist_ok=True)
        cv.imwrite(f'{debug_dir}/gray_{os.path.basename(fname)}', gray)
        
        # Find checkerboard corners
        print(f"Attempting to find {checkerboard_size[0]}x{checkerboard_size[1]} corners...")
        ret, corners = cv.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            print("✓ Found corners!")
            successful_images += 1
            
            # Refine corner positions to sub-pixel accuracy
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            # Store the points
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            # Save the successful detection image
            debug_img = img.copy()
            cv.drawChessboardCorners(debug_img, checkerboard_size, corners2, ret)
            cv.imwrite(f'{debug_dir}/success_{os.path.basename(fname)}', debug_img)
        else:
            print("✗ No corners detected in this image")
            # Save the failed image with a different prefix for debugging
            debug_img = img.copy()
            cv.putText(debug_img, "Failed Detection", (30, 30), 
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.imwrite(f'{debug_dir}/failed_{os.path.basename(fname)}', debug_img)
    
    print(f"\nSuccessfully processed {successful_images} out of {len(images)} images")
    
    if successful_images < 10:
        print("Warning: For reliable calibration, at least 10 successful images are recommended")
        if successful_images == 0:
            return None, None, None, None
    
    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, 
                                                     image_size, None, None)
    
    # Calculate re-projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    total_error = mean_error/len(objpoints)
    print(f"\nCalibration complete! Average re-projection error: {total_error}")
    
    # Return the camera matrix and distortion coefficients
    return mtx, dist, image_size, total_error

def save_calibration(mtx, dist, image_size, error, filename='camera_calibration.json'):
    # Convert numpy arrays to lists for JSON serialization
    calibration_data = {
        'camera_matrix': mtx.tolist(),
        'dist_coeffs': dist.tolist(),
        'image_size': image_size,
        'reprojection_error': error
    }
    
    with open(filename, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    print(f"\nCalibration parameters saved to {filename}")

# Use the functions
if __name__ == "__main__":
    image_dir = "images_for_calibration/300_300"
    
    # Adjust these to match your checkerboard pattern
    checkerboard_size = (10, 7)  # internal corners, not squares
    
    mtx, dist, image_size, error = calibrate_camera(image_dir,
                                                    checkerboard_size)
    if mtx is not None:
        save_calibration(mtx, dist, image_size, error)