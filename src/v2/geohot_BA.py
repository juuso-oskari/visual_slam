from pickle import TRUE
import numpy as np
import g2o
from map import *
#import cv2
"""
def localBundleAdjustement(BA, KeyFrames):
    # Loop over all the keyframes
    for frame in KeyFrames:
        # Add every keyframe to posegraph
        BA.add_pose(frame.ID, frame.pose)
        # Loop over all 3d points that the frame sees
        for landmark_id in frame.landmarks.keys: # points3d is a dictionary where key is id and value is list of xyz point and original detection point in image 
            point_xyz, image_point = frame.landmarks[landmark_id]
            BA.add_point(landmark_id, point_xyz)
            BA.add_edge(landmark_id, frame.ID, image_point)
"""


class BundleAdjustment2(g2o.SparseOptimizer):
    def __init__(self, camera):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        # TODO: understand how to get focal length from x y
        self.focal_length = (camera.fx, camera.fy)
        self.principal_point = (camera.cx, camera.cy)
        self.baseline = 0 # TODO: figure this out
        self.fx, self.fy = camera.fx, camera.fy
        self.cx, self.cy = camera.cx, camera.cy
        g2o.VertexSCam.set_cam(*self.focal_length, *self.principal_point, self.baseline)
        #super().add_parameter(cam)

    def optimize(self, max_iterations=10, verbose=True):
        super().initialize_optimization()
        #super().set_verbose(verbose)
        super().optimize(max_iterations)
        
    def save_to_file(self, filename):
        super().save(filename)
    
    '''
    def add_pose(self, pose_id, pose, fixed=False):
        se3 = g2o.SE3Quat(pose[0:3,0:3], pose[0:3,3])
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(se3)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)
        # TODO: check pose correctness 
    '''
    
    def add_pose(self, pose_id, pose, fixed=False):
        p = g2o.Isometry3d(pose)
        v_se3 = g2o.VertexSCam()
        v_se3.set_id(pose_id*2)
        v_se3.set_estimate(p)
        v_se3.set_fixed(fixed)
        v_se3.set_all()
        super().add_vertex(v_se3)
    
    
    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        if self.vertex(point_id * 2 + 1) == None:
            super().add_vertex(v_p)


    def add_edge(self, point_id, pose_id, 
            measurement, edge_id,
            information=np.identity(2)*0.5,
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI
        # geohot 
        #edge = g2o.EdgeSE3ProjectXYZ()
        edge = g2o.Edge_XYZ_VSC()
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        #edge.set_cam(*self.focal_length, *self.principal_point, self.baseline)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        edge.set_parameter_id(0, 0)
        super().add_edge(edge)

    
    def add_edge_between_poses(self, parent_id, child_id, measurement, information=np.eye(6), robust_kernel=g2o.RobustKernelDCS()):
        edge = g2o.EdgeSE3()
        vertices = [parent_id, child_id]
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v*2)
            edge.set_vertex(i, v)

        #edge.set_vertex(0, self.vertex(parent_id ))
        #edge.set_vertex(1, self.vertex(child_id ))
        
        edge.set_measurement(g2o.Isometry3d(measurement))  # relative pose transformation between frames
        edge.set_information(information)
        edge.set_parameter_id(0,0)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)
    
    
    
    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()

    #  The local bundle adjustment refines the pose of the current key frame, the poses of connected key frames, and all the map points observed in these key frames.
    # So we need to update only poses of the last two keyframes added to map (last_keyframe_id), and the 3d points seen by these two keyframes
    def localBundleAdjustement(self, map, last_keyframe_id=None, scale=False):
        frame_ids = map.frames.keys()
        point_ids = map.points_3d.keys()
        # if last_keyframe_id argument is given, do only local
        if last_keyframe_id != None:
            frame_ids = [last_keyframe_id-1, last_keyframe_id]
            point_ids = map.GetPointsVisibleToFrames(frame_ids)
        
        for frame_id in frame_ids:
            frame_obj = map.GetFrame(frame_id)
            if(frame_id== 0):
                self.add_pose(pose_id=frame_id, pose = frame_obj.GetPose(), fixed=True) # set initial frame as fixed (origo)
            else:
                self.add_pose(pose_id=frame_id, pose = frame_obj.GetPose())
                #for parent_ID in frame_obj.GetParentIDs():
                    # add edge between parent and current frame (usually previous and current frame, with loop closure as exception)
                    # self.add_edge_between_poses(parent_id = parent_ID, child_id = frame_id, measurement=frame_obj.GetTransitionWithParentID(parent_ID))
                    
            
        for point_id in point_ids:
            point_obj = map.GetPoint(point_id)
            self.add_point(point_id=point_id, point=point_obj.Get3dPoint(), fixed=False)
            # add edge between point and frames that see the point
            for frame_id in frame_ids:
                correspondence = point_obj.GetFrame(frame_id)
                if correspondence != None:
                    _, uv, _ =  correspondence
                    self.add_edge(point_id=point_id, pose_id=frame_id, measurement=uv,  edge_id=point_id*frame_id+10000000)
            
            #for frame, uv, descriptor in point_obj.frames:
            #    self.add_edge(point_id=point_id, pose_id=frame.GetID(), measurement=uv,  edge_id=point_id*frame.GetID()+10000000)
        # run the optimization
        self.save_to_file("before.g2o")
        self.optimize()
        self.save_to_file("after.g2o")
        median_depth = 1.0
        if scale:
            vector_norms = []
            for point_id in point_ids:
                vector_norms.append(np.linalg.norm(self.get_point(point_id)))
            median_depth = np.median(np.array(vector_norms))
        # update map
        for frame_id in frame_ids:
            new_pose = self.get_pose(frame_id).matrix()
            new_pose[0:3,3] /= median_depth
            map.UpdatePose(new_pose = new_pose, frame_id = frame_id)
        for point_id in point_ids:
            map.UpdatePoint3D(new_point = self.get_point(point_id)/median_depth, point_id = point_id)
            
        
    # differs from localBundleAdjustement by setting map points as fixed and estimating motion (poses) only
    # if init is set to True, adds the points in the map to graph (should be done only once)
    def motionOnlyBundleAdjustement(self, map, scale=False, save=False):
        frame_ids = map.frames.keys()
        point_ids = map.points_3d.keys()
        for frame_id in frame_ids:
            frame_obj = map.GetFrame(frame_id)
            if(frame_obj.IsKeyFrame()):
                self.add_pose(pose_id=frame_id, pose = frame_obj.GetPose(), fixed=True) # set key frame as fixed
            else:
                self.add_pose(pose_id=frame_id, pose = frame_obj.GetPose())
                #for parent_ID in frame_obj.GetParentIDs():
                    # add edge between parent and current frame (usually previous and current frame, with loop closure as exception)
                    #self.add_edge_between_poses(parent_id = parent_ID, child_id = frame_id, measurement=frame_obj.GetTransitionWithParentID(parent_ID))
        for point_id in point_ids:
            point_obj = map.GetPoint(point_id)
            self.add_point(point_id=point_id, point=point_obj.Get3dPoint(), fixed=True) # add all global map points as fixed as this is motion only
            for frame_id in frame_ids:
                correspondence = point_obj.GetFrame(frame_id)
                if correspondence != None:
                    _, uv, _ =  correspondence
                    self.add_edge(point_id=point_id, pose_id=frame_id, measurement=uv,  edge_id=point_id*frame_id+10000000)

        # run the optimization
        self.optimize()
        median_depth = 1
        if scale:
            vector_norms = []
            for point_id in point_ids:
                vector_norms.append(np.linalg.norm(self.get_point(point_id)))
            median_depth = np.median(np.array(vector_norms))

        # update local map
        for frame_id in frame_ids:
            new_pose = self.get_pose(frame_id).matrix()
            new_pose[0:3,3] /= median_depth
            map.UpdatePose(new_pose = new_pose, frame_id = frame_id)


        