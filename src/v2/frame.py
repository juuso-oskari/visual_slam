import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self):
        self.extractor = cv2.SIFT_create()
        
    def compute_features(self, img):
        pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
        kp, des = self.extractor.compute(img, kps)
        return kp, des

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

    def AddPose(self, id, init_pose):
        self.pose = init_pose
        self.ID = id
    
    def UpdatePose(self, new_pose):
        self.pose = new_pose
    
    def UpdateLandmark(self, landmarkID, new_xyzPoint):
        self.landmarks[landmarkID] = [new_xyzPoint, self.landmarks[landmarkID][1]]        
        
    def GetAll3dPoints(self):
        points3d = []
        for key in self.landmarks:
            points3d.append(self.landmarks[key][0])
        return np.squeeze(np.array(points3d))