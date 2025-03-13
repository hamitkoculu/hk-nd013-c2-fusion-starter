# Writeup: Track 3D-Objects Over Time

Please use this starter template to answer the following questions:

### You can see the pictures in the pc_examples file.

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?

    I implemented a Kalman filter for state estimation, track management for initializing and deleting objects, data association using a nearest-neighbor approach, and camera-lidar fusion to improve accuracy. The most challenging part was data association due to handling occlusions and sensor noise.


### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 

    Yes, camera-lidar fusion improves object classification and tracking robustness by combining depth and appearance information. In my results, fusion helped with better tracking in occluded and cluttered environments.


### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?

    Challenges include sensor misalignment, time synchronization, and environmental conditions affecting sensor performance. In the project, I observed slight misalignment and occasional mismatches in data association.


### 4. Can you think of ways to improve your tracking results in the future?

    Future improvements could include using a more advanced association algorithm like the Hungarian method, incorporating deep learning for feature extraction, and refining sensor calibration techniques.