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
from viewer import Viewer
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
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
    def match_features(self, frame_cur, frame_prev):
        kp1, desc1 = frame_cur.keypoints, frame_cur.features
        kp2, desc2 = frame_prev.keypoints, frame_prev.features
        # Match descriptors.
        matches = self.matcher.knnMatch(desc1,desc2,k=1)
        # Sort the matches according to nearest neighbor distance ratio (NNDR) (CV course, exercise 4)
        distmat = np.dot(desc1, desc2.T)
        X_terms = np.expand_dims(np.diag(np.dot(desc1, desc1.T)), axis=1)
        X_terms = np.tile(X_terms,(1,desc2.shape[0]))
        Y_terms = np.expand_dims(np.diag(np.dot(desc2, desc2.T)), axis=0)
        Y_terms = np.tile(Y_terms,(desc1.shape[0],1))
        distmat = np.sqrt(Y_terms + X_terms - 2*distmat)
        ## We determine the mutually nearest neighbors
        dist1 = np.amin(distmat, axis=1)
        ids1 = np.argmin(distmat, axis=1)
        dist2 = np.amin(distmat, axis=0)
        ids2 = np.argmin(distmat, axis=0)
        pairs = []
        for k in range(ids1.size):
            if k == ids2[ids1[k]]:
                pairs.append(np.array([k, ids1[k], dist1[k]]))
        pairs = np.array(pairs)
        # We sort the mutually nearest neighbors based on the nearest neighbor distance ratio
        NNDR = []
        for k,ids1_k,dist1_k in pairs:
            r_k = np.sort(distmat[int(k),:])
            nndr = r_k[0]/r_k[1]
            NNDR.append(nndr)

        id_nnd = np.argsort(NNDR)
        return np.array(matches)[id_nnd]


class Frame:
    def __init__(self, rgb_fp, d_path, feature_extractor):
        self.rgb = cv2.imread(rgb_fp)
        self.depth = cv2.imread(d_path)
        self.keypoints, self.features  = None, None
        self.feature_extractor = feature_extractor
    def process_frame(self):
        self.keypoints, self.features = self.feature_extract(self.rgb)
        return self.keypoints, self.features, self.rgb
        
    def feature_extract(self, rgb):
        return self.feature_extractor.compute_features(rgb)

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
    poses = [np.eye(4)]
    # run feature extraction for 1st image
    fp_rgb = dir_rgb + str(1) + ".png"
    fp_depth = dir_depth + str(1) + ".png"
    cur_frame = Frame(fp_rgb, fp_depth, feature_extractor)
    kp, features, rgb = cur_frame.process_frame() 
    prev_frame = cur_frame
    
    for i in range(2,1000):
        if i % 20 == 0:
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
                pass # continue
            # https://www.programcreek.com/python/example/70413/cv2.RANSAC
            # match and normalize keypoints
            #preMatchedPoints, curMatchedPoints = MatchAndNormalize(prev_frame.keypoints, cur_frame.keypoints, matches, K)
            preMatchedPoints, curMatchedPoints = MatchPoints(prev_frame.keypoints, cur_frame.keypoints, matches)
            # compute homography and inliers
            H, inliersH, scoreH  = estimateHomography(preMatchedPoints, curMatchedPoints, homTh=4.0) # ransac threshold as last argument
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
            R,t, validFraction = estimateRelativePose(tform, inlierPrePoints[::2], inlierCurrPoints[::2], K, tform_type)
            if(validFraction < 0.9):
                pass #continue
            # according to https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
            #RelativePoseTransformation = np.linalg.inv(np.vstack((np.hstack((R,t[:,np.newaxis])), np.array([0,0,0,1]))))
            RelativePoseTransformation = Isometry3d(R=R, t=np.squeeze(t)).inverse().matrix()
            pose = RelativePoseTransformation @ poses[-1]
            viewer.update_pose(pose = g2o.Isometry3d(pose))
            poses.append(pose)
            new_xyz = trajectory[-1] + pose[:3,3]
            trajectory.append(new_xyz)
            # TODO: triangulate two view to obtain 3-D map points
            
            
            # Display
            #img3 = cv2.drawMatchesKnn(prev_frame.rgb,prev_frame.keypoints, cur_frame.rgb,cur_frame.keypoints,matches[:100],None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #img2 = cv2.drawKeypoints(rgb, kp, None, color=(0,255,0), flags=0)
            #cv2.imshow('a', img3)
            #cv2.waitKey(1)
            prev_frame = cur_frame
    viewer.stop()
    
    
