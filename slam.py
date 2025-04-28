# Install necessary packages
!pip install -q breezyslam opencv-python matplotlib

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from breezyslam.algorithms import TinySlam
from breezyslam.sensors import Laser

# Step 1: Load the 2D room map image
# (Make sure you have uploaded 'room_map.png' or generate it with the earlier code)
map_img = cv2.imread('/mnt/data/room_map.png', cv2.IMREAD_GRAYSCALE)

# Convert to binary map: 0 = wall, 1 = free space
binary_map = np.where(map_img > 127, 1, 0)

height, width = binary_map.shape

# Step 2: Define Robot and Lidar Parameters
robot_pos = np.array([100, 100])  # Start at center
robot_angle = 0  # Facing right (0 degrees)

# Lidar Parameters
LIDAR_ANGLE_RES = 2  # degrees between rays
LIDAR_MAX_RANGE = 100  # maximum range in pixels
SPEED = 2  # robot speed (pixels per step)

# SLAM setup
class FakeLaser(Laser):
    def __init__(self):
        super().__init__('FakeLaser', scan_size=360//LIDAR_ANGLE_RES, scan_rate_hz=10, detection_angle_degrees=360, distance_no_detection_mm=LIDAR_MAX_RANGE*10, offset_mm=0)

laser = FakeLaser()
slam = TinySlam(laser, width, height)
mapbytes = bytearray(width * height)

trajectory = []

# Step 3: Simulate Lidar Scanning
def simulate_lidar(pos, angle):
    scan = []
    for ray_angle in range(0, 360, LIDAR_ANGLE_RES):
        total_angle = np.deg2rad(angle + ray_angle)
        for r in range(0, LIDAR_MAX_RANGE, 1):
            x = int(pos[0] + r * np.cos(total_angle))
            y = int(pos[1] + r * np.sin(total_angle))
            if x < 0 or x >= width or y < 0 or y >= height:
                scan.append(r * 10)
                break
            if binary_map[y, x] == 0:
                scan.append(r * 10)  # Convert to mm
                break
        else:
            scan.append(LIDAR_MAX_RANGE * 10)
    return scan

# Step 4: Move Robot and Update SLAM
# Robot simple auto-move (random walk)

import random

moves = ['forward', 'left', 'right']

for step in range(300):  # simulate 300 steps
    scan = simulate_lidar(robot_pos, robot_angle)
    
    # Feed scan into SLAM
    slam.update(scan, int(robot_pos[0]*10), int(robot_pos[1]*10))
    
    # Save position
    trajectory.append(robot_pos.copy())
    
    # Random movement
    move = random.choice(moves)
    if move == 'forward':
        new_pos = robot_pos + SPEED * np.array([np.cos(np.deg2rad(robot_angle)), np.sin(np.deg2rad(robot_angle))])
    elif move == 'left':
        robot_angle = (robot_angle - 30) % 360
        new_pos = robot_pos
    else:  # right
        robot_angle = (robot_angle + 30) % 360
        new_pos = robot_pos
    
    # Check collision
    x, y = int(new_pos[0]), int(new_pos[1])
    if 0 <= x < width and 0 <= y < height and binary_map[y, x] == 1:
        robot_pos = new_pos  # move if no wall

# Step 5: Visualize Trajectory and Map

trajectory = np.array(trajectory)

plt.figure(figsize=(10,10))

# Plot original room map
plt.subplot(1,2,1)
plt.title('Original Room Map')
plt.imshow(binary_map, cmap='gray')
plt.plot(trajectory[:,0], trajectory[:,1], 'r-')
plt.axis('off')

# Plot SLAM generated map
slam.getmap(mapbytes)

slam_map = np.array(mapbytes, dtype=np.uint8).reshape((height, width))

plt.subplot(1,2,2)
plt.title('SLAM Generated Map')
plt.imshow(slam_map, cmap='gray')
plt.plot(trajectory[:,0], trajectory[:,1], 'g-')
plt.axis('off')

plt.show()
