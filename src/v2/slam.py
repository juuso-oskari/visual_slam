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

if __name__=="__main__":
    D = np.array([0, 0, 0, 0], dtype=np.float32)  # no distortion
    K = np.matrix([[481.20, 0, 319.5], [0, 480.0, 239.5], [0, 0, 1]])  # camera intrinsic parameters
    fx, fy, cx, cy = 481.20, 480.0, 319.5, 239.5
    # Filepaths
    cur_dir = "/home/juuso"
    dir_rgb = cur_dir + "/visual_slam/data/ICL_NUIM/rgb/"
    dir_depth = cur_dir + "/visual_slam/data/ICL_NUIM/depth/"
    fp_rgb = dir_rgb + str(1) + ".png"
    fp_depth = dir_depth + str(1) + ".png"

    # Initializations of classes
    viewer = Viewer()
    feature_extractor = FeatureExtractor()
    feature_matcher = FeatureMatcher()
    map = Map()
    id_frame = 0 # even numbers for the poses
    id_point = 1 # odd numbers for the 3d points
    # Process first frame
    cur_frame = Frame(fp_rgb, fp_depth, feature_extractor, id=id_frame)
    cur_frame.AddPose(init_pose=np.eye(4)) # initial pose (identity) and id
    cur_frame.SetAsKeyFrame()   # Set initial frame to be keyframe
    cur_frame.AddParent(None) # set parent to None for initial frame
    kp, features, rgb = cur_frame.process_frame() # Process frame features etc

    # Add inital frame to map
    map.AddFrame(frame_id=id_frame, frame=cur_frame)
    id_frame = id_frame + 1

    #cur_frame.UpdatePose(np.eye(4)*100)
    #print(cur_frame.GetPose())
    #print(map.GetFrame(id_frame).GetPose())
    
    for i in range(2,1200):
        
        fp_rgb = dir_rgb + str(i) + ".png"
        fp_depth = dir_depth + str(i) + ".png"

        # Feature Extraction for current frame
        prev_frame = map.GetFrame(id_frame-1) # Get previous frame from the map class
        cur_frame = Frame(fp_rgb, fp_depth, feature_extractor, id=id_frame)
        kp, features, rgb = cur_frame.process_frame() 
        matches = feature_matcher.match_features(prev_frame, cur_frame)
        if( len(matches) < 100 ) :
            continue
        # Match corresponding image points
        preMatchedPoints, curMatchedPoints = MatchPoints(prev_frame.GetKeyPoints(), cur_frame.GetKeyPoints(), matches)
        ## compute essential and inliers
        E, inliers , score = estimateEssential(preMatchedPoints, curMatchedPoints, K, essTh=3.0 / K[0,0])
        # Leave only inliers based on geometric verification
        inlierPrePoints = preMatchedPoints[inliers[:, 0] == 1, :]
        inlierCurrPoints = curMatchedPoints[inliers[:, 0] == 1, :]
        # get pose transformation (use only half of the points for faster computation)
        R, t, validFraction, triangulatedPoints, inlierPrePoints, inlierCurrPoints = estimateRelativePose(E, inlierPrePoints[::2], inlierCurrPoints[::2], K, "Essential")
        if(validFraction < 0.9):
            continue
        # according to https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
        RelativePoseTransformation = Isometry3d(R=R, t=np.squeeze(t)).inverse().matrix()
        # Calculate current frame pose in world coordinate system
        pose = RelativePoseTransformation @ prev_frame.GetPose()
        # Add edges
        prev_frame.AddChild(cur_frame) # Add Child frame to previous frame
        cur_frame.AddParent(prev_frame) # Add prev frame as a parent to current frame
        cur_frame.AddPose(init_pose=pose)
        # Transform triagualted points to world frame
        pts_objs = (np.linalg.inv(prev_frame.GetPose()) @ triangulatedPoints).T
        pts_objs = pts_objs[:,:3] / np.asarray(pts_objs[:,-1]).reshape(-1,1)
        # Update viewer
        viewer.update_pose(pose = g2o.Isometry3d(pose), cloud = pts_objs, colour=np.array([[0],[0],[0]]).T)
        # reaches this point only when new keyframe is found
        # -> so add the triangulated point objects to map
        for pt, uv1, uv2 in zip(pts_objs, inlierPrePoints, inlierCurrPoints):
            pt_object = Point(location=pt, id=id_point)
            pt_object.AddFrame(frame=prev_frame, uv=uv1)
            pt_object.AddFrame(frame=cur_frame, uv=uv2)
            map.AddPoint3D(point_id=id_point, point_3d=pt_object) # add to map
            id_point = id_point + 1  
        cur_frame.SetAsKeyFrame()        
        break
    
    
    viewer.stop()
    # local bundleadjustement
    camera = Camera(fx,fy,cx,cy)
    BA = BundleAdjustment(camera)
    BA.localBundleAdjustement(map)
    
    
    
    
    
    
    
    