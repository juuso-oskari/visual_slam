import numpy as np
import cv2
from copy import deepcopy

class FeatureExtractor:
    def __init__(self):
        self.extractor = cv2.SIFT_create()
        
    def compute_features(self, img):
        pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
        kp, des = self.extractor.compute(img, kps)
        return cv2.KeyPoint_convert(kp), des

class FeatureMatcher():
    def __init__(self):
        self.matcher = cv2.BFMatcher()

    def match_features(self, kp1, desc1, kp2, desc2, ratio = 0.8):

        # Match descriptors.
        rawMatches = self.matcher.knnMatch(desc1,desc2,k=2)
        # perform Lowe's ratio test to get actual matches
        matches = []
        pts1 = [] # matched image points in img1
        pts2 = [] # matched image points in img2
        ft1 = [] # matched features in img1
        ft2 = [] # matched features in img2
        for m, n in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < ratio * n.distance:
                # here queryIdx corresponds to kpsA
                # trainIdx corresponds to kpsB
                pts2.append(kp2[m.trainIdx])
                pts1.append(kp1[m.queryIdx])
                #pts1.append(kp1[m.queryIdx])
                #pts2.append(kp2[m.trainIdx])
                ft1.append(desc1[m.queryIdx])
                ft2.append(desc2[m.trainIdx])
                matches.append([m])
        
        pts1  = np.asarray(pts1)
        pts2 = np.asarray(pts2)
        ft1  = np.asarray(ft1)
        ft2 = np.asarray(ft2)
        
        return matches, pts1, ft1, pts2, ft2

class Frame:
    def __init__(self, rgb_fp, d_path, id):
        # Image related attributes
        self.rgb = cv2.imread(rgb_fp)
        self.depth = cv2.imread(d_path)
        self.keypoints, self.features  = None, None
        #self.feature_extractor = feature_extractor

        # Camera related attributed
        self.ID = id 
        self.pose = None # Pose estimation might fail for some frames
        # TODO: The following should also store the transitions between frames
        # Parent frames
        self.parents = {} # key: parent_frame_id, value : transition from parent to current frame
        # Child frames
        self.childs = []
        # keyframe flag, determines if the 
        self.keyframe = False

    # AddParent adds parent frame to frame class. Parent frame is added as {parent_id : transition between parent and current frame} key-value pair
    def AddParent(self, parent_frame_id, transition):
        self.parents[parent_frame_id] = transition
    
    def GetParentIDs(self):
        return self.parents.keys()
    
    def GetTransitionWithParentID(self, parent_id):
        return self.parents[parent_id]
        
    # Processes the frame by calling feature_extract method
    def process_frame(self, feature_extractor):
        self.keypoints, self.features = self.feature_extract(self.rgb, feature_extractor)
        return self.keypoints, self.features, self.rgb

    # Extracts features by calling method compute_features that is implemeted in class FeatureExtractor
    def feature_extract(self, rgb, feature_extractor):
        return feature_extractor.compute_features(rgb)

    # Adds initial pose. ie this function adds pose that corresponds to this frame (Before any optimization),
    def AddPose(self, init_pose):
        self.pose = init_pose
    
    # Pose is here updated by the pose that is obtained from optimization.
    def UpdatePose(self, new_pose):
        self.pose = new_pose


    # AddChild adds child frame to this frame.
    def AddChild(self, child_frame):
        self.childs.append(child_frame)
    
    def GetPose(self):
        return self.pose

    def SetAsKeyFrame(self):
        self.keyframe = True

    def GetKeyPoints(self):
        return self.keypoints

    def GetFeatures(self):
        return self.features
    
    def GetID(self):
        return self.ID
    
    def IsKeyFrame(self):
        return self.keyframe
    
    def AddID(self, new_id):
        self.ID = new_id

