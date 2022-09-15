import numpy as np
import cv2
import time
import os 
from pathlib import Path
import re

class FeatureExtractor:
    def __init__(self):
        self.extractor = cv2.ORB_create()
    def compute_features(self, img):
        kp, des = self.extractor.detectAndCompute(img,None)
        return kp, des


class FeatureMatcher():
    def __init__(self):
        #self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.matcher = cv2.BFMatcher()
    def match_features(self, frame_cur, frame_prev):
        kp1, feat1 = frame_cur.keypoints, frame_cur.features
        kp2, feat2 = frame_prev.keypoints, frame_prev.features
        # Match descriptors.
        matches = self.matcher.knnMatch(feat1,feat2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.01*n.distance:
                good.append([m])
        # Sort them in the order of their distance.
        #matches = sorted(matches, key = lambda x:x.distance)
        return good





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
        
        

if __name__=="__main__":
    # Filepaths
    cur_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    dir_rgb = cur_dir + "/visual_slam/data/ICL_NUIM/rgb/"
    dir_depth = cur_dir + "/visual_slam/data/ICL_NUIM/depth/"
    is_WINDOWS = False
    if is_WINDOWS:
        dir_rgb = dir_rgb.replace("/", "\\")
        dir_depth = dir_depth.replace("/", "\\")
    # Initialize
    feature_extractor = FeatureExtractor()
    feature_matcher = FeatureMatcher()
    fp_rgb = dir_rgb + str(1) + ".png"
    fp_depth = dir_depth + str(1) + ".png"
    frame = Frame(fp_rgb, fp_depth, feature_extractor)
    kp, features, rgb = frame.process_frame() 
    # Display
    img2 = cv2.drawKeypoints(rgb, kp, None, color=(0,255,0), flags=0)
    cv2.imshow('a', img2)
    cv2.waitKey(0)
    #
    prev_frame = frame

    for i in range(2,100):
        fp_rgb = dir_rgb + str(i) + ".png"
        fp_depth = dir_depth + str(i) + ".png"
        # Feature Extraction for current frame
        cur_frame = Frame(fp_rgb, fp_depth, feature_extractor)
        kp, features, rgb = cur_frame.process_frame()
        # Feature Matching to previous frame
        matches = feature_matcher.match_features(cur_frame, prev_frame)    
        # Display
        #img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img3 = cv2.drawMatchesKnn(prev_frame.rgb,prev_frame.keypoints,cur_frame.rgb,cur_frame.keypoints,matches,None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #img2 = cv2.drawKeypoints(rgb, kp, None, color=(0,255,0), flags=0)
        cv2.imshow('a', img3)
        cv2.waitKey(0)
        #
        prev_frame = cur_frame
