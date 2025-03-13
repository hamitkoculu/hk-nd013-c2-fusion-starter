# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import zlib
import open3d as o3d

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")

    # Step 1: Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Step 2: Convert the numpy point cloud to Open3D format
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])  # Use only XYZ coordinates

    # Step 3: Use height (Z) as color
    z_min, z_max = np.min(pcl[:, 2]), np.max(pcl[:, 2])  # Get min and max height
    z_colors = (pcl[:, 2] - z_min) / (z_max - z_min)  # Normalize height to [0,1]
    colors = np.c_[z_colors, np.zeros_like(z_colors), 1 - z_colors]  # Map to blue-red colormap
    
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Step 4: Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Lidar Point Cloud")

    return

    #######
    ####### ID_S1_EX2 END #######     


# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # Step 1: Extract lidar data
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    
    # Step 2: Extract range image
    ri = dataset_pb2.MatrixFloat()
    ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
    ri = np.array(ri.data).reshape(ri.shape.dims)
    
    # Step 3: Set values <0 to zero
    ri[ri < 0] = 0.0
    
    # Step 4: Convert range channel to 8-bit
    ri_range = ri[:, :, 0]
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
    img_range = ri_range.astype(np.uint8)
    
    # Step 5: Convert intensity channel to 8-bit (using 1-99 percentile normalization)
    ri_intensity = ri[:, :, 1]
    min_intensity = np.percentile(ri_intensity, 1)
    max_intensity = np.percentile(ri_intensity, 99)
    ri_intensity = np.clip(ri_intensity, min_intensity, max_intensity)
    ri_intensity = 255 * (ri_intensity - min_intensity) / (max_intensity - min_intensity)
    img_intensity = ri_intensity.astype(np.uint8)
    
    # Step 6: Crop range image to +/- 90 degrees
    deg90 = int(img_intensity.shape[1] / 2)
    ri_center = int(img_intensity.shape[1] / 2)
    img_range = img_range[:, ri_center - deg90:ri_center + deg90]
    img_intensity = img_intensity[:, ri_center - deg90:ri_center + deg90]

    # Step 7: Stack images vertically
    img_range_intensity = np.vstack((img_range, img_intensity)).astype(np.uint8)

    # Step 8: Visualize with OpenCV
    cv2.imshow("Range Image", img_range_intensity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs, vis=False):
    # Remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # Shift ground level to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  
    
    # Convert sensor coordinates to BEV-map coordinates
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")
    
    ## Step 1: Compute BEV-map discretization
    bev_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    
    ## Step 2: Transform x-coordinates to BEV-image coordinates    
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret))
    
    ## Step 3: Transform y-coordinates and shift to avoid negative values
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret) + (configs.bev_width + 1) / 2)
    
    ## Step 4: Visualize point-cloud
    show_pcl(lidar_pcl_cpy)
    ####### ID_S2_EX1 END #######     
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")
    
    ## Step 1: Create intensity map filled with zeros
    intensity_map = np.zeros((configs.bev_height, configs.bev_width))
    
    ## Step 2: Sort points by x, y, and -z coordinates
    idx_sorted = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_top = lidar_pcl_cpy[idx_sorted]
    
    ## Step 3: Extract unique x, y points and keep only top-most z values
    _, idx_indices, _ = np.unique(lidar_pcl_top[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_pcl_top = lidar_pcl_top[idx_indices]
    
    ## Step 4: Normalize and assign intensity values
    min_intensity = np.percentile(lidar_pcl_top[:, 3], 1)
    max_intensity = np.percentile(lidar_pcl_top[:, 3], 99)
    lidar_pcl_top[:, 3] = np.clip(lidar_pcl_top[:, 3], min_intensity, max_intensity)

    intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 3] / (np.amax(lidar_pcl_top[:, 3])-np.amin(lidar_pcl_top[:, 3]))
    
    ## Step 5: Visualize intensity map
    img_intensity = intensity_map * 256
    img_intensity = img_intensity.astype(np.uint8)
    cv2.imshow('img_intensity', img_intensity) 
    ####### ID_S2_EX2 END ####### 
    
    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")
    
    ## Step 1: Create height map filled with zeros
    height_map = np.zeros((configs.bev_height, configs.bev_width))
    
    ## Step 2: Normalize and assign height values
    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    
    ## Step 3: Visualize height map
    img_height = height_map * 256
    img_height = img_height.astype(np.uint8)
    cv2.imshow('height_map', height_map) 
    ####### ID_S2_EX3 END #######    
    
    




    # TODO remove after implementing all of the above steps
#    lidar_pcl_cpy = []
#    lidar_pcl_top = []
#    height_map = []
#    intensity_map = []

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps




