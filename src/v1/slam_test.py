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
from viewer import Viewer
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
#import pandas as pd



import matplotlib.pyplot as plt

from viewer import Viewer

class FeatureExtractor:
    def __init__(self):
        self.extractor = cv2.SIFT_create()
        
    def compute_features(self, img):
        pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
        kp, des = self.extractor.compute(img, kps)
        return kp, des
        
        #kp, des = self.extractor.detectAndCompute(img,None)
        #return kp, des


class FeatureMatcher():
    def __init__(self):
        self.matcher = cv2.BFMatcher()
    def match_features(self, frame_prev, frame_cur, ratio = 0.8):
        kp1, desc1 = frame_prev.keypoints, frame_prev.features
        kp2, desc2 = frame_cur.keypoints, frame_cur.features
        # Match descriptors.
        rawMatches = self.matcher.knnMatch(desc1,desc2,k=2)
        # perform Lowe's ratio test to get actual matches
        matches = []
        for m, n in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < ratio * n.distance:
                # here queryIdx corresponds to kpsA
                # trainIdx corresponds to kpsB
                matches.append([m])
        return matches

class Frame:
    def __init__(self, rgb_fp, d_path, feature_extractor):
        self.rgb = cv2.imread(rgb_fp)
        self.depth = cv2.imread(d_path)
        self.keypoints, self.features  = None, None
        self.feature_extractor = feature_extractor
        self.ID, self.pose = None, None
        self.landmarks = {}
        
    def process_frame(self):
        self.keypoints, self.features = self.feature_extract(self.rgb)
        return self.keypoints, self.features, self.rgb
        
    def feature_extract(self, rgb):
        return self.feature_extractor.compute_features(rgb)

    def StoreLandmark(self, landmarkID, xyzPoint, imagePoint):
        self.landmarks[landmarkID] = [xyzPoint, imagePoint]

    def AddPose(self, id, pose):
        self.pose = pose
        self.ID = id



class Point:
    def __init__(self, xyz):
        self.xyz = xyz
        self.projections = {} # (key,value)-pairs, key: pose (ID) where this point is visible from, value: projection to image plane
    def AddProjection(self, poseID, kp):
        self.projections[poseID] = kp
    def GetProjection(self, poseID):
        return self.projections[poseID]


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
    # Global variables
    debug = False
    scale = 5000
    D = np.array([0, 0, 0, 0], dtype=np.float32)  # no distortion
    K = np.matrix([[481.20, 0, 319.5], [0, 480.0, 239.5], [0, 0, 1]])  # camera intrinsic parameters
    fx, fy, cx, cy = 481.20, 480.0, 319.5, 239.5
    # Filepaths
    cur_dir = "/home/juuso"
    dir_rgb = cur_dir + "/visual_slam/data/ICL_NUIM/rgb/"
    dir_depth = cur_dir + "/visual_slam/data/ICL_NUIM/depth/"
    is_WINDOWS = False
    if is_WINDOWS:
        dir_rgb = dir_rgb.replace("/", "\\")
        dir_depth = dir_depth.replace("/", "\\")
    # Initialize
    viewer = Viewer()
    feature_extractor = FeatureExtractor()
    feature_matcher = FeatureMatcher()
    trajectory = [np.array([0, 0, 0])] # camera trajectory for visualization
    #trajectory2 = np.array([0, 0, 0]) # camera trajectory for visualization
    poses = [np.eye(4)] # inverses of point transforms, opencv by default gives point transforms between images as to where the points move instead of camera moving
    # run feature extraction for 1st image
    fp_rgb = dir_rgb + str(1) + ".png"
    fp_depth = dir_depth + str(1) + ".png"
    cur_frame = Frame(fp_rgb, fp_depth, feature_extractor)
    kp, features, rgb = cur_frame.process_frame() 
    prev_frame = cur_frame
    map = []
    KeyFrames = [cur_frame]
    
    # Map initialization
    for i in range(2,1200):
        if i % 1 == 0:
            print(i)
            fp_rgb = dir_rgb + str(i) + ".png"
            fp_depth = dir_depth + str(i) + ".png"
            # Feature Extraction for current frame
            cur_frame = Frame(fp_rgb, fp_depth, feature_extractor)
            kp, features, rgb = cur_frame.process_frame()
            # Feature Matching to previous frame
            matches = feature_matcher.match_features(prev_frame, cur_frame)    
            # if not enough matches (<100) continue to next frame
            if(len(matches) < 100):
                print("too few matches")
                continue # continue
            
            # https://www.programcreek.com/python/example/70413/cv2.RANSAC
            # match and normalize keypoints
            # CAUTION: normalizing or not normalizing change the results so be careful
            #preMatchedPoints, curMatchedPoints = MatchAndNormalize(prev_frame.keypoints, cur_frame.keypoints, matches, K)
            preMatchedPoints, curMatchedPoints = MatchPoints(prev_frame.keypoints, cur_frame.keypoints, matches)
            # compute homography and inliers
            #H, inliersH, scoreH  = estimateHomography(preMatchedPoints, curMatchedPoints, homTh=4.0) # ransac threshold as last argument
            scoreH = 0
            ## compute essential and inliers
            E, inliersE , scoreE = estimateEssential(preMatchedPoints, curMatchedPoints, K, essTh=3.0 / K[0,0])
            if debug:
                print("Homography score: ")
                print(scoreH)
                print("Essential score: ")
                print(scoreE)
                print("T with Homography: ")
                R,t, validFraction = estimateRelativePose(H, preMatchedPoints[inliersH[:, 0] == 1, :], curMatchedPoints[inliersH[:, 0] == 1, :], K, "Homography")
                print(t)
                print("T with Essential: ")
                R,t, validFraction = estimateRelativePose(E, preMatchedPoints[inliersE[:, 0] == 1, :], curMatchedPoints[inliersE[:, 0] == 1, :], K, "Essential")
                print(t)
                print("T with recoverPose(Essential): ")
                points, R, t, inliers = cv2.recoverPose(E, preMatchedPoints[inliersE[:, 0] == 1, :], curMatchedPoints[inliersE[:, 0] == 1, :], cameraMatrix=K)
                print(t)
            
            # TODO: Select the model based on a heuristic
            ratio = scoreH/(scoreH + scoreE)
            ratioThreshold = 0.45
            if ratio > ratioThreshold:
                inliers = inliersH
                tform = H
                tform_type = "Homography"
                print("Chose homography")
            else:
                inliers = inliersE
                tform = E
                tform_type = "Essential"
                print("Chose essential")
            # currently selects essential everytime    
            #inliers = inliersE
            #tform = E
            #tform_type = "Essential" 
            # else continue with the inliers
            inlierPrePoints = preMatchedPoints[inliers[:, 0] == 1, :]
            inlierCurrPoints = curMatchedPoints[inliers[:, 0] == 1, :]
            # get pose transformation (use only half of the points for faster computation)
            R, t, validFraction, triangulatedPoints, inlierPrePoints, inlierCurrPoints = estimateRelativePose(tform, inlierPrePoints[::2], inlierCurrPoints[::2], K, tform_type)
            if(validFraction < 0.9):
                continue
            # according to https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
            PointTransformation = Isometry3d(R=R, t=np.squeeze(t)).matrix()
            RelativePoseTransformation = Isometry3d(R=R, t=np.squeeze(t)).inverse().matrix()
            pose = RelativePoseTransformation @ poses[-1]
            poses.append(pose)
            
            pts_obj = (np.linalg.inv(poses[-2]) @ triangulatedPoints).T
            pts_obj = pts_obj[:,:3] / np.asarray(pts_obj[:,-1]).reshape(-1,1)
            viewer.update_pose(pose = g2o.Isometry3d(pose), cloud = pts_obj, colour=np.array([[0],[0],[0]]).T)
            map.append(pts_obj)
            # Display
            #img3 = cv2.drawMatchesKnn(prev_frame.rgb,prev_frame.keypoints, cur_frame.rgb,cur_frame.keypoints,matches,None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #img2 = cv2.drawKeypoints(rgb, kp, None, color=(0,255,0), flags=0)
            #cv2.imshow('a', img3)
            #cv2.waitKey(0)
            prev_frame = cur_frame
            break
        
    # Initialize BundleAdjustement
    camera = Camera(fx,fy,cx,cy)
    BA = BundleAdjustment(camera)
    
    
    KeyFrames.append(cur_frame)
    Pose_id = 1 #"pose".encode('utf-8').hex() + hex(1)
    KeyFrames[0].AddPose(Pose_id, poses[0])
    Pose_id += 1
    KeyFrames[1].AddPose(Pose_id, poses[-1])
    
    
    landmark_id = 20 # "landmark".encode('utf-8').hex() + hex(1)
    print(np.shape(triangulatedPoints))
    for point3d, imagepoint1, imagepoint2 in zip(pts_obj, inlierPrePoints, inlierCurrPoints):
        KeyFrames[-2].StoreLandmark(landmark_id, xyzPoint=point3d, imagePoint=imagepoint1)
        KeyFrames[-1].StoreLandmark(landmark_id, xyzPoint=point3d, imagePoint=imagepoint2)
        landmark_id += 1 # hex(1)
    
    
    
    BA.localBundleAdjustement(KeyFrames)
    
    
    
    viewer.stop()
    
    
    
    

    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #print(map)
    # Data for a three-dimensional line
    for p,m in zip(poses,map):
        if (np.min(m[:,0]) <-1000 or np.max(m[:,0]) >1000):
            continue
        if (np.min(m[:,2]) <-1000 or np.max(m[:,2]) >1000):
            continue
        if (np.min(m[:,1]) <-1000 or np.max(m[:,1]) >1000):
            continue
            #print(p[0,3])
        ax.scatter(m[:,0],m[:,1],m[:,2],c='blue',marker="o")
        ax.scatter(p[0,3], p[1,3], p[2,3], c='red', marker="p")
    plt.show()
    """
    
