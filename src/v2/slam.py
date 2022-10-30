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
    
    for i in range(1,1200):
        fp_rgb = dir_rgb + rgb_images[i] #str(i) + ".png"
        fp_depth = dir_depth + str(i) + ".png"
        # Feature Extraction for current frame
        prev_frame = map.GetFrame(id_frame-1) # Get previous frame from the map class
        cur_frame = Frame(fp_rgb, fp_depth, id=id_frame)
        kp_cur, features_cur, rgb_cur = cur_frame.process_frame(feature_extractor=feature_extractor) 
        # pts1, ft1, pts2, ft2
        matches,  preMatchedPoints, preMatchedFeatures, curMatchedPoints, curMatchedFeatures = feature_matcher.match_features(kp_prev, features_prev, kp_cur, features_cur)
        if( len(matches) < 100 ) :
            continue 
        # Match corresponding image points
        #preMatchedPoints, curMatchedPoints = MatchPoints(prev_frame.GetKeyPoints(), cur_frame.GetKeyPoints(), matches)
        #preMatchedFeatures, curMatchedFeatures = MatchFeatures(prev_frame.GetFeatures(), cur_frame.GetFeatures(), matches)
        ## compute essential and inliers
        E, inliers , score = estimateEssential(preMatchedPoints, curMatchedPoints, K, essTh=3.0 / K[0,0])
        # Leave only inliers based on geometric verification
        inlierPrePoints = preMatchedPoints[inliers[:, 0] == 1, :]
        inlierPreFeatures = preMatchedFeatures[inliers[:, 0] == 1, :]    
        inlierCurrPoints = curMatchedPoints[inliers[:, 0] == 1, :]
        inlierCurrFeatures = curMatchedFeatures[inliers[:, 0] == 1, :]
        # get pose transformation (use only half of the points for faster computation)
        # SOlution by the gooods
        R, t, validFraction, triangulatedPoints, inlierPrePoints, inlierCurrPoints, inlierPreFeatures, inlierCurrFeatures = estimateRelativePose(E, inlierPrePoints, inlierCurrPoints, inlierPreFeatures, inlierCurrFeatures, K, "Essential")
        
        if(validFraction < 0.9):
            continue
        # according to https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
        RelativePoseTransformation = Isometry3d(R=R, t=np.squeeze(t)).inverse().matrix()
        # Calculate current frame pose in world coordinate system
        pose = RelativePoseTransformation @ prev_frame.GetPose()
        # Add edges
        prev_frame.AddChild(child_frame=cur_frame) # Add Child frame to previous frame
        cur_frame.AddParent(parent_frame_id=prev_frame.GetID(), transition=RelativePoseTransformation) # Add prev frame as a parent to current frame
        cur_frame.AddPose(init_pose=pose) # Add pose calculated to the current frame
        map.AddFrame(frame_id=id_frame, frame=cur_frame) # Add current frame to the map
        id_frame = id_frame + 1
        # Transform triagualted points to world frame
        pts_objs = (np.linalg.inv(prev_frame.GetPose()) @ triangulatedPoints).T
        pts_objs = pts_objs[:,:3] / np.asarray(pts_objs[:,-1]).reshape(-1,1) 
        # Update viewer
        viewer.update_pose(pose = g2o.Isometry3d(pose), cloud = pts_objs, colour=np.array([[0],[0],[0]]).T)
        # reaches this point only when new keyframe is found
        # -> so add the triangulated point objects to map
        
        for pt, uv1, uv2, ft1, ft2 in zip(pts_objs, inlierPrePoints, inlierCurrPoints, inlierPreFeatures, inlierCurrFeatures):
            pt_object = Point(location=pt, id=id_point) # Create point class with 3d point and point id
            pt_object.AddFrame(frame=prev_frame, uv=uv1, descriptor=ft1) # Add first frame to the point object. This is frame where the point was detected
            pt_object.AddFrame(frame=cur_frame, uv=uv2, descriptor=ft2)# Add second frame to the point object. This is frame where the point was detected
            map.AddPoint3D(point_id=id_point, point_3d=pt_object) # add to map
            id_point = id_point + 1  # Increment point id
        cur_frame.SetAsKeyFrame()  # Sets cur frame as the keyframe

        break
        
    print(i)
    
    viewer.stop()
    # local bundleadjustement
    
    camera = Camera(fx,fy,cx,cy)
    BA = BundleAdjustment(camera)
    
    BA.localBundleAdjustement(map, scale=True) # update global map
    # visualize the initialized map
    viewer2 = Viewer()
    map.visualize_map(viewer2)
    
    # store last keyframe as copy (we do not want to change it during tracking)
    last_keyframe = deepcopy(map.GetFrame(frame_id=id_frame-1))
    # Start local tracking mapping process
    # For every new tracking period, create a clean local map, which only updates poses using motionOnlyBundleAdjustement()
    # Once a new key frame is detected, update global map
    local_map = Map()
    local_map.AddFrame(last_keyframe.GetID(), last_keyframe)
    # start local frame indexing
    print("last keyframe id: ", last_keyframe.GetID())
    id_frame_local = id_frame
    # add to local map the points from global map, which the last keyframe sees
    local_map.Store3DPoints(map.GetCopyOfPointObjects(last_keyframe.GetID()))
    # Store current pose into pose
    pose = last_keyframe.GetPose()
    
    loop_idx = i
    for i in range(loop_idx, 150):
        print("Image index: ", i)
        # features are extracted for each new frame
        # and then matched (using matchFeatures), with features in the last key frame
        # that have known corresponding 3-D map points. 
        fp_rgb = dir_rgb + rgb_images[i] #str(i) + ".png"
        fp_depth = dir_depth + str(i) + ".png"
        cur_frame = Frame(fp_rgb, fp_depth, id=id_frame_local)
        kp_cur, features_cur, rgb_cur = cur_frame.process_frame(feature_extractor=feature_extractor) # This returns keypoints as numpy.ndarray
        # Get keypoints and features in the last key frame corresponding to known 3D-points
        kp_prev, features_prev, known_3d = local_map.GetImagePointsWithFrameID(last_keyframe.GetID()) # This returns keypoints as numpy.ndarray
        matches,  preMatchedPoints, preMatchedFeatures, curMatchedPoints, curMatchedFeatures = feature_matcher.match_features(kp_prev, features_prev, kp_cur, features_cur)
        # get matched 3d locations
        known_3d = np.array([known_3d[m[0].queryIdx] for m in matches])
        # TODO: possibly give previous rvec and tvec as initial guesses
        #retval, rvec, tvec, inliers = cv2.solvePnPRansac(known_3d_matched, curMatchedPoints, K, D, confidence=0.95, iterationsCount=10000, reprojectionError=3) #, confidence=0.999, reprojectionError=3.0/K[0,0])
        #success, rotation_vector, translation_vector = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist_coeffs, flags=0)
        row_of_ones = np.ones_like(known_3d[:,0])[np.newaxis,:]
        known_3d_in_previous_frame = (pose @ np.concatenate( (known_3d.T, row_of_ones ), axis=0)).T[:,0:3]
        print(np.shape(known_3d_in_previous_frame))
        success, rvec, tvec = cv2.solvePnP(objectPoints=known_3d_in_previous_frame, imagePoints=curMatchedPoints, cameraMatrix=K, distCoeffs=D)
        T = transformMatrix(rvec, tvec)
        r, t = T[:3, :3], np.asarray(T[:3, -1]).squeeze()
        RelativePoseTransformation = Isometry3d(R=r, t=t).inverse().matrix()
        pose = RelativePoseTransformation @ pose
        # Add edges
        print("local frame id", id_frame_local)
        local_map.GetFrame(frame_id=id_frame_local-1).AddChild(child_frame=cur_frame) # Add cur frame as child to previous frame
        cur_frame.AddParent(parent_frame_id=local_map.GetFrame(frame_id=id_frame_local-1).GetID(), transition=RelativePoseTransformation) # Add parent as previous frame ID : relative pose transformation key-value pair
        cur_frame.AddPose(init_pose=pose) # Add pose calculated to the current frame
        local_map.AddFrame(frame_id=id_frame_local, frame=cur_frame) # Add current frame to the map
        id_frame_local = id_frame_local + 1
        # Do motion only bundle adjustement with local map
        BA.motionOnlyBundleAdjustement(local_map, scale=True)
        local_map.visualize_map(viewer=viewer2)
        #viewer2.update_pose(pose = g2o.Isometry3d(pose), cloud = None, colour=np.array([[0],[0],[0]]).T)
        img3 = cv2.drawMatchesKnn(last_keyframe.rgb, Numpy2Keypoint(kp_prev), rgb_cur, Numpy2Keypoint(kp_cur), matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('a', img3)
        cv2.waitKey(1)

    viewer2.stop()
    print("Ruljhati")
    loop_idx = i
    
    
    
    
    
    
    
    
    
    
    