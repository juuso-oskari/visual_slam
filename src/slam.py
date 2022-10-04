import numpy as np
import cv2
import time
import os 
from pathlib import Path
import re
   
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


class FeatureMatcher:
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
    K = np.matrix([[481.20, 0, 319.5], [0, 480.0, 239.5], [0, 0, 1]])  # camera intrinsic parameters
    # run feature extraction for 1st image
    fp_rgb = dir_rgb + str(1) + ".png"
    fp_depth = dir_depth + str(1) + ".png"
    cur_frame = Frame(fp_rgb, fp_depth, feature_extractor)
    kp, features, rgb = cur_frame.process_frame() 
    prev_frame = cur_frame

    for i in range(2,1000):
        if i % 30 == 0:
            fp_rgb = dir_rgb + str(i) + ".png"
            fp_depth = dir_depth + str(i) + ".png"
            # Feature Extraction for current frame
            cur_frame = Frame(fp_rgb, fp_depth, feature_extractor)
            kp, features, rgb = cur_frame.process_frame()
            # Feature Matching to previous frame
            matches = feature_matcher.match_features(cur_frame, prev_frame) 
            # Display
            img3 = cv2.drawMatchesKnn(cur_frame.rgb,cur_frame.keypoints,prev_frame.rgb,prev_frame.keypoints,matches[:100],None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #img2 = cv2.drawKeypoints(rgb, kp, None, color=(0,255,0), flags=0)
            cv2.imshow('a', img3)
            cv2.waitKey(0)
            #
            prev_frame = cur_frame
