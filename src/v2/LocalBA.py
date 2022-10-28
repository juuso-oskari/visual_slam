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


class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, camera):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        # TODO: understand how to get focal length from x y
        f = (camera.fx + camera.fy) / 2.0
        cam = g2o.CameraParameters(f, (camera.cx, camera.cy), 0)
        cam.set_id(0)
        super().add_parameter(cam)

    def optimize(self, max_iterations=10, verbose=True):
        super().initialize_optimization()
        super().set_verbose(verbose)
        super().optimize(max_iterations)
        

    def add_pose(self, pose_id, pose, fixed=False):
        se3 = g2o.SE3Quat(pose[0:3,0:3], pose[0:3,3])
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(se3)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)
        # TODO: check pose correctness 

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        if self.vertex(point_id * 2 + 1) == None:
            super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)
        edge.set_parameter_id(0,0)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()

    def localBundleAdjustement(self, map):
        frame_ids = map.frames.keys()
        point_ids = map.points_3d.keys()
        for frame_id in frame_ids:
            frame_obj = map.GetFrame(frame_id)
            if(frame_id== 0):
                self.add_pose(pose_id=frame_id, pose = frame_obj.GetPose(), fixed=True) # set initial frame as fixed (origo)
            else:
                self.add_pose(pose_id=frame_id, pose = frame_obj.GetPose())
        for point_id in point_ids:
            point_obj = map.GetPoint(point_id)
            self.add_point(point_id=point_id, point=point_obj.Get3dPoint())
            for frame, uv, descriptor in point_obj.frames:
                self.add_edge(point_id=point_id, pose_id=frame.GetID(), measurement=uv)

        # run the optimization
        self.optimize()
        # update map
        for frame_id in frame_ids:
            map.UpdatePose(new_pose = self.get_pose(frame_id).matrix(), frame_id = frame_id)
        for point_id in point_ids:
            map.UpdatePoint3D(new_point = self.get_point(point_id), point_id = point_id)
            
        
        
                
        