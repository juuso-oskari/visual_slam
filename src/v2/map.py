import numpy as np
import cv2


class Map:
    def __init__(self):
        self.frames = {}
        self.points_3d = {}

    def AddFrame(self, frame_id, frame):
        # TODO add warning if adding duplicate
        if frame_id in self.frames.keys():
            raise Exception("Duplicate frame warning")
        self.frames[frame_id] = frame 

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
        if point_id not in self.frames.keys():   
            self.points_3d[point_id].UpdatePoint(new_point)
        else:
            raise Exception("No frame yet added") 
    
    def GetFrame(self, frame_id):
        return self.frames[frame_id]
    
    def GetPoint(self, point_id):
        return self.points_3d[point_id]

        