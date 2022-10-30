import numpy as np
import cv2

class Point:
    def __init__(self, location, id):
        self.ID = id
        self.frames = [] # list of tuples. (frame, uv). Where uv is the 2d location on the image plane of the frame
        self.location_3d = location

    def AddFrame(self, frame, uv, descriptor):
        self.frames.append((frame, uv, descriptor))


    def UpdatePoint(self, new_location):
        self.location_3d = new_location

    # checks if frame with frame_id sees this point
    def IsVisibleTo(self, frame_id):
        for frame, uv, descriptor in self.frames:
            if (frame_id == frame.ID):
                return True
        return False
    
    # Gets image point (2d) based on frame id
    def GetImagePoint(self, frame_id):
        for frame, uv, descriptor in self.frames:
            if (frame_id == frame.ID):
                return (uv, descriptor)
            
    def Get3dPoint(self):
        return self.location_3d
    
    def GetVectorNorm(self):
        return np.linalg.norm(self.location_3d)