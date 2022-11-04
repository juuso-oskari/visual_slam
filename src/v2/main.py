import imp
import numpy as np
import g2o
import cv2
from helper_functions import *
import pangolin
import OpenGL.GL as gl
import time
import os 
from pathlib import Path
import re
from LocalBA import BundleAdjustment
from frame import *
from point import *
from map import Map
from viewer import Viewer
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from copy import deepcopy
import graphslam
np.set_printoptions(suppress=True)

class Camera:
    def __init__(self, fx, fy, cx, cy, baseline=1):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline

class Isometry3d(object):
    """3d rigid transform."""
    def __init__(self, R, t):
        self.R = R
        self.t = t
    def matrix(self):
        m = np.identity(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m
    def inverse(self):
        return Isometry3d(self.R.T, -self.R.T @ self.t)
    def __mul__(self, T1):
        R = self.R @ T1.R
        t = self.R @ T1.t + self.t
        return Isometry3d(R, t)   
    def orientation(self):
        return self.R
    def position(self):
        return self.t


def MapInitialization()


def main():
    D = np.array([0, 0, 0, 0], dtype=np.float32)  # no distortion
    #K = np.matrix([[481.20, 0, 319.5], [0, 480.0, 239.5], [0, 0, 1]])  # camera intrinsic parameters
    #fx, fy, cx, cy = 481.20, 480.0, 319.5, 239.5
    K = np.matrix([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])  # camera intrinsic parameters
    fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
    
    # Filepaths
    rgb_images = os.listdir("data/rgbd_dataset_freiburg3_long_office_household/rgb")
    rgb_images.sort(key=lambda f: int(re.sub('\D', '', f)))
    cur_dir = "/home/juuso"
    dir_rgb = cur_dir + "/visual_slam/data/rgbd_dataset_freiburg3_long_office_household/rgb/"
    dir_depth = cur_dir + "/visual_slam/data/ICL_NUIM/depth/"
    fp_rgb = dir_rgb + rgb_images[0] #str(1) + ".png"
    print(fp_rgb)
    fp_depth = dir_depth + str(1) + ".png"

    # Initializations of classes
    viewer = Viewer()
    feature_extractor = FeatureExtractor()
    feature_matcher = FeatureMatcher()
    map = Map()
    id_frame = 0 # even numbers for the poses
    id_point = 1 # odd numbers for the 3d points
    # Process first frame
    cur_frame = Frame(fp_rgb, fp_depth, id=id_frame)
    cur_frame.AddPose(init_pose=np.eye(4)) # initial pose (identity) and id
    cur_frame.SetAsKeyFrame()   # Set initial frame to be keyframe
    cur_frame.AddParent(None, None) # set parent to None for initial frame
    kp_prev, features_prev, rgb_prev = cur_frame.process_frame(feature_extractor=feature_extractor) # Process frame features etc

    # Add inital frame to map
    map.AddFrame(frame_id=id_frame, frame=cur_frame)
    id_frame = id_frame + 1
    