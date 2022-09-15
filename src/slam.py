import numpy as np
import cv2
import time
import os 
from pathlib import Path

class FeatureExtractor:
    def __init__(self):
        self.extractor = cv2.ORB_create()
    def compute_features(self, img):
        kp, des = self.extractor.detectAndCompute(img,None)
        return kp, des



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

    cur_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    dir_rgb = cur_dir + "/visual_slam/data/ICL_NUIM/rgb/"
    dir_depth = cur_dir + "/visual_slam/data/ICL_NUIM/depth/"
    is_WINDOWS = False
    if is_WINDOWS:
        dir_rgb = cur_dir + "\visual_slam\data\ICL_NUIM\rgb\"
        dir_depth = cur_dir + "\visual_slam\data\ICL_NUIM\depth\"

    feature_extractor = FeatureExtractor()

    for i in range(1,100):
        fp_rgb = dir_rgb + str(i) + ".png"
        fp_depth = dir_depth + str(i) + ".png"
        frame = Frame(fp_rgb, fp_depth, feature_extractor)
        kp, features, rgb = frame.process_frame()
        img2 = cv2.drawKeypoints(rgb, kp, None, color=(0,255,0), flags=0)
        cv2.imshow('a', img2)
        cv2.waitKey(0)
