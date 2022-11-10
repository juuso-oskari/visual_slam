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
from geohot_BA import *
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
    cur_dir = "/home/jere"
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
        # Add frame to map, add parent, and relative pose between  these two
        map.AddParentAndPose(parent_id = id_frame-1, frame_id = id_frame, frame_obj = cur_frame, rel_pose_trans = RelativePoseTransformation, pose = pose)
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
    
    loop_idx = i # continue where map initialization left off
    print("last keyframe idx", i)
    

    for i in range(loop_idx+1, 1000):
        print("Image index: ", i)
        # features are extracted for each new frame
        # and then matched (using matchFeatures), with features in the last key frame
        # that have known corresponding 3-D map points. 
        fp_rgb = dir_rgb + rgb_images[i] #str(i) + ".png"
        fp_depth = dir_depth + str(i) + ".png"
        cur_frame = Frame(fp_rgb, fp_depth, id=id_frame_local)
        kp_cur, features_cur, rgb_cur = cur_frame.process_frame(feature_extractor=feature_extractor) # This returns keypoints as numpy.ndarray
        # Get keypoints and features in the last key frame corresponding to known 3D-points
        kp_prev, features_prev, known_3d, point_IDs = local_map.GetImagePointsWithFrameID(last_keyframe.GetID()) # This returns keypoints as numpy.ndarray
        print("\nTracking " + str(len(known_3d)) + " points\n")
        matches,  preMatchedPoints, preMatchedFeatures, curMatchedPoints, curMatchedFeatures = feature_matcher.match_features(kp_prev, features_prev, kp_cur, features_cur)
        # get 3d locations and their ids in map of feature points matched in the new frame
        known_3d_matched = np.array([known_3d[m[0].queryIdx] for m in matches])
        known_3d_matched_ids = [point_IDs[m[0].queryIdx] for m in matches]
        
        # TODO: possibly give previous rvec and tvec as initial guesses
        W_T_prev = local_map.GetFrame(id_frame_local-1).GetPose()
        prev_T_W = Isometry3d(R=W_T_prev[0:3,0:3], t=np.asarray(W_T_prev[:3, -1]).squeeze()).inverse().matrix()
        rvec_guess = Rtorvec(W_T_prev[0:3,0:3]) # use previous estimates as initial guess to help in computational efficiency
        tvec_guess = W_T_prev[0:3,3]
        #print("RÄNSÄCKKI lasketaan N pisteen perusteella", len(known_3d_matched))
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=known_3d_matched[:,np.newaxis,:].astype(np.float32), imagePoints=curMatchedPoints[:,np.newaxis,:].astype(np.float32), cameraMatrix=K, distCoeffs=np.array([]),
                                                        rvec=rvec_guess.copy(), tvec=tvec_guess.copy(), useExtrinsicGuess=True)#, rvec=rvec_guess, tvec=tvec_guess, useExtrinsicGuess=True)
        #retval, rvec, tvec = cv2.solvePnP(objectPoints=known_3d_matched[:,np.newaxis,:].astype(np.float32), imagePoints=curMatchedPoints[:,np.newaxis,:].astype(np.float32), cameraMatrix=K, distCoeffs=np.array([]))
        tvec = tvec[:,np.newaxis]
        
        T = transformMatrix(rvec, tvec)
        r, t = T[:3, :3], np.asarray(T[:3, -1]).squeeze()
        W_T_curr = Isometry3d(R=r, t=t).inverse().matrix() # form wold frame to current camera frame
        RelativePoseTransformation = prev_T_W @ W_T_curr

        # Else continue tracking by adding to local map
        # Add frame to map, add parent, and relative pose between  these two
        local_map.AddParentAndPose(parent_id = id_frame_local-1, frame_id = id_frame_local, frame_obj = cur_frame, rel_pose_trans = RelativePoseTransformation, pose = W_T_curr)
        # Add point to frame connection for the new frame in each local map point
        local_map.AddPointToFrameCorrespondences(point_ids = [point_IDs[m[0].queryIdx] for m in matches], image_points = curMatchedPoints, descriptors = curMatchedFeatures, frame_obj = cur_frame)
        
        # TODO: Do motion only bundle adjustement with local map
        localBA = BundleAdjustment(camera)
        localBA.motionOnlyBundleAdjustement(local_map, scale=False, save=True)
        #viewer2.update_pose(pose = g2o.Isometry3d(local_map.GetFrame(id_frame_local).GetPose()), cloud = None, colour=np.array([[0],[0],[0]]).T)
        viewer2.update_image(cv2.resize(cv2.drawMatchesKnn(last_keyframe.rgb, Numpy2Keypoint(kp_prev), rgb_cur, Numpy2Keypoint(kp_cur), matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS), None, fx=.6, fy=.6))
        # increment local frame id
        # Check if current frame is a key frame:
        # 1. at least 20 frames has passed or current frame tracks less than 80 map points
        # 2. The map points tracked are fewer than 90% of the map points seen by the last key frame
        if (i-loop_idx > 20 or len(curMatchedPoints) < 80) and (len(curMatchedPoints) < 0.9*len(known_3d)):
            loop_idx = i
            cur_frame.SetAsKeyFrame()
            W_T_prev_key = map.GetFrame(id_frame-1).GetPose() # Get last keyframe pose from global map
            prev_key_T_W = Isometry3d(R=W_T_prev_key[0:3,0:3], t=np.asarray(W_T_prev_key[:3, -1]).squeeze()).inverse().matrix()
            W_T_cur_key = local_map.GetFrame(id_frame_local).GetPose() # get optimized pose from local map
            # Clear tracking frame from the parents
            cur_frame.ClearParent()
            # Update global map by adding new keyframe 
            map.AddParentAndPose(parent_id = id_frame-1, frame_id = id_frame, frame_obj = cur_frame, rel_pose_trans = prev_key_T_W @ W_T_cur_key, pose = W_T_cur_key)
            # Add Point frame correspondance
            map.AddPointToFrameCorrespondences(point_ids = known_3d_matched_ids, image_points = curMatchedPoints, descriptors = curMatchedFeatures, frame_obj = cur_frame)
            # Remove outlier map points that are observed in fewer than 3 key frames
            #map.DiscardOutlierMapPoints(n_visible_frames=3)
            # TODO: Feature mathing between frames frame_id-1 and frame_id with unmatched points ie not with points that are already in the map
            image_points_already_in_map = map.GetImagePointsWithFrameID(id_frame-1)[0] # this contains matched (=mapped) image points
            
            kp1 = map.GetFrame(id_frame-1).GetKeyPoints() # this contains all the image points (matched and unmatched)
            desc1 = map.GetFrame(id_frame-1).GetFeatures()
            idx = GetListDiff(kp1, image_points_already_in_map) # get unmatched points from previous keyframe, TODO: should it be keyframes
            kp1 = kp1[idx]
            desc1 = desc1[idx]
            matches,  last_keyframe_points, last_keyframe_features, cur_keyframe_points, cur_keyframe_features = feature_matcher.match_features(kp1 = kp1, 
                                                                                    desc1= desc1, kp2 = map.GetFrame(id_frame).GetKeyPoints(), desc2 = map.GetFrame(id_frame).GetFeatures())
            # TODO: Triagulate matches and add those to map 
            
            """
            p1 = Isometry3d(R=map.GetFrame(id_frame-1).GetPose()[0:3, 0:3], t=np.asarray(map.GetFrame(id_frame-1).GetPose()[:3, -1]).squeeze()).matrix()
            p2 = Isometry3d(R=map.GetFrame(id_frame).GetPose()[0:3, 0:3], t=np.asarray(map.GetFrame(id_frame).GetPose()[:3, -1]).squeeze()).matrix()
            P1 = CameraProjectionMatrix(R = p1[0:3, 0:3], t = p1[:3, 3:], K = K)
            P2 = CameraProjectionMatrix(R = p2[0:3, 0:3], t = p2[:3, 3:], K = K)
            P1 = CameraProjectionMatrix2(map.GetFrame(id_frame-1).GetPose(), K)
            P2 = CameraProjectionMatrix2(map.GetFrame(id_frame).GetPose(), K)
            
            x1 = MakeHomogeneous(last_keyframe_points)
            x2 = MakeHomogeneous(cur_keyframe_points)
            """
           
            # poses of two keyframes (4x4 matrices) : from World to Camera Frame
            #p1 = map.GetFrame(id_frame-1).GetPose()
            #p2 = map.GetFrame(id_frame).GetPose()
            p1 = Isometry3d(R=map.GetFrame(id_frame-1).GetPose()[0:3, 0:3], t=np.asarray(map.GetFrame(id_frame-1).GetPose()[:3, -1]).squeeze()).inverse().matrix()
            p2 = Isometry3d(R=map.GetFrame(id_frame).GetPose()[0:3, 0:3], t=np.asarray(map.GetFrame(id_frame).GetPose()[:3, -1]).squeeze()).inverse().matrix()
            # projection matrices (3x4)
            #Proj1 = CameraProjectionMatrix(R = p1[0:3, 0:3], t = p1[:3, 3:], K = K)
            #Proj2 = CameraProjectionMatrix(R = p2[0:3, 0:3], t = p2[:3, 3:], K = K)
            Proj1 = CameraProjectionMatrix2(Pose=p1, K=K)
            Proj2 = CameraProjectionMatrix2(Pose=p2, K=K)
            # triangulated points (Nx4) 
            x1 = MakeHomogeneous(last_keyframe_points) # matched image points in map.GetFrame(id_frame-1)
            x2 = MakeHomogeneous(cur_keyframe_points) # matched image points in map.GetFrame(id_frame)
            
            
            #new_triagulated_points = triangulate_points_course(P1 = Proj1, P2 = Proj2, x1s = x1, x2s = x2)
        
            
            
            # TODO: Fix triangulate, most likely a scaling issue, since new points are drawn as rays on concecutive updates of viewer (same point different scaling)
            # Cpp style
            #new_triagulated_points = triangulateCpp(proj1 = p1, proj2 = p2, pts1 = x1, pts2 = x2)
            #new_triagulated_points /= new_triagulated_points[:,3:]
            # Geohot style
            new_triagulated_points = triangulate(pose1 = Proj1, pose2 = Proj2, pts1 = x1, pts2 = x2)
            new_triagulated_points /= new_triagulated_points[:,3:]
            # OpenCV style
            #new_triagulated_points = cv2.triangulatePoints(projMatr1=(Proj1).astype(np.float), projMatr2=(Proj2).astype(np.float), projPoints1=(x1[:,:2].T).astype(np.float), projPoints2=(x2[:,:2].T).astype(np.float))
            #new_triagulated_points /= new_triagulated_points[3]
            #new_triagulated_points = new_triagulated_points.T
            
            proj1 = p1 @ new_triagulated_points.T
            proj2 = p2 @ new_triagulated_points.T
            
            proj_img1 = Proj1 @ new_triagulated_points.T
            proj_img2 = Proj2 @ new_triagulated_points.T
            proj_img2 /= proj_img2[2]
            
            print("Projecting back to new camera frame (not image plane)")
            print(proj2.T[:10])
            
            print("Projecting to image plane")
            print(proj_img2.T[:10])
            print("Original matches in image plane")
            print(x2[:10])
            
            
            new_triagulated_points = new_triagulated_points[:,:3]
            # cherilarity check (positive depth when projected)
            good_points_idx =  np.where( (proj1[2] > 0) & (proj2[2] > 0) & (proj2[2] < 1) & (proj1[2] < 1))[0] #range(len(new_triagulated_points))#np.where( (proj1[2] > 0) & (proj2[2] > 0) & (proj2[2] < 1) & (proj1[2] < 1))[0]
            # just for visualization
            new_points_ids = []
            for pt, uv1, uv2, ft1, ft2 in zip(new_triagulated_points[good_points_idx], last_keyframe_points[good_points_idx], cur_keyframe_points[good_points_idx], last_keyframe_features[good_points_idx], cur_keyframe_features[good_points_idx]):
                pt_object = Point(location=pt, id=id_point) # Create point class with 3d point and point id
                pt_object.AddFrame(frame=map.GetFrame(id_frame-1), uv=uv1, descriptor=ft1) # Add first frame to the point object. This is frame where the point was detected
                pt_object.AddFrame(frame=map.GetFrame(id_frame), uv=uv2, descriptor=ft2)# Add second frame to the point object. This is frame where the point was detected
                map.AddPoint3D(point_id=id_point, point_3d=pt_object) # add to map
                new_points_ids.append(id_point)
                id_point = id_point + 1  # Increment point id
                
            
            # TODO: Bundle adjustement
            BA = BundleAdjustment(camera)
            BA.localBundleAdjustement(map)
            # update viewer with new pose and new points (both optimized)
            #print(np.shape(map.GetAll3DPoints()))
            #viewer.stop()
            #viewer = Viewer()
            viewer2.update_pose(pose = g2o.Isometry3d(map.GetFrame(id_frame).GetPose()), cloud = map.GetAll3DPoints(), colour=np.array([[0],[0],[0]]).T)

            # Store as last keyframe and increment indices correctly (local tracking starts again at next index)

            # store last keyframe as copy (we do not want to change it during tracking)
            last_keyframe = deepcopy(map.GetFrame(frame_id=id_frame))
            # Start local tracking mapping process again
            # For every new tracking period, create a clean local map, which only updates following poses using motionOnlyBundleAdjustement()
            local_map = Map()
            local_map.AddFrame(last_keyframe.GetID(), last_keyframe)
            # start local frame indexing

            print("last keyframe id: ", last_keyframe.GetID())
            id_frame = id_frame + 1
            id_frame_local = id_frame

            # add to local map the points from global map, which the last keyframe sees
            local_map.Store3DPoints(map.GetCopyOfPointObjects(last_keyframe.GetID()))
        
        else: # else just continue local tracking
            id_frame_local = id_frame_local + 1
           
    #map.visualize_map(viewer=viewer2)

    viewer2.stop()
    print("End of run")
    loop_idx = i
    
    
    
    
    
    
    
    
    
    
    