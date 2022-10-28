import numpy as np
import cv2
import g2o

class Map:
    def __init__(self):
        self.frames = {}
        self.points_3d = {}

    def AddFrame(self, frame_id, frame):
        # TODO add warning if adding duplicate
        if frame_id in self.frames.keys():
            raise Exception("Duplicate frame warning")
        self.frames[frame_id] = frame 


    # Get image points and corresponding descriptors
    # Returns list of tuples (uv, feat)
    def GetImagePointsWithFrameID(self, frame_id):
        image_points = []
        descriptors = []
        locations_3d = []
        for point_obj in self.points_3d.values():
            # get image points from tuple
            image_points.append(point_obj.GetImagePoint(frame_id)[0])  
            # get descriptor from tuple
            descriptors.append(point_obj.GetImagePoint(frame_id)[1]) 
            # get the known 3d location
            locations_3d.append(point_obj.Get3dPoint()) 
        return np.array(image_points), np.array(descriptors), np.array(locations_3d)
    
    
    def GetAll3DPoints(self):
        allpoints = []
        for point_obj in self.points_3d.values():
            allpoints.append(point_obj.location_3d)
        return np.array(allpoints).reshape(-1,3)
    
    def GetAllPoses(self):
        allposes = []
        for frame_obj in self.frames.values():
            allposes.append(frame_obj.GetPose())
        return allposes
    
    def AddPoint3D(self, point_id, point_3d):
        if point_id in self.points_3d.keys():
            raise Exception("Duplicate point3d warning")
        self.points_3d[point_id] = point_3d

    def UpdatePose(self, new_pose, frame_id):
        if frame_id in self.frames.keys():
            self.frames[frame_id].UpdatePose(new_pose)
        else:
            raise Exception("No frame yet added")

    def UpdatePoint3D(self, new_point, point_id):
        if point_id in self.points_3d.keys():   
            self.points_3d[point_id].UpdatePoint(new_point)
        else:
            raise Exception("No point yet added") 
    
    def GetFrame(self, frame_id):
        return self.frames[frame_id]
    
    def GetPoint(self, point_id):
        return self.points_3d[point_id]

    def visualize_map(self, viewer):
        i = 0
        for pose in self.GetAllPoses():
            if i==0:
                viewer.update_pose(pose = g2o.Isometry3d(pose), cloud = self.GetAll3DPoints(), colour=np.array([[0],[0],[0]]).T)
            else:
                viewer.update_pose(pose = g2o.Isometry3d(pose), colour=np.array([[0],[0],[0]]).T)
            i += 1 # draw points only for the first iteration