import numpy as np
import cv2

class Point:
    def __init__(self, location, id):
        self.ID = id
        #self.frames = [] # list of triples. (frame, uv, descriptor). Where uv is the 2d location on the image plane of the frame, and descriptor is e.g SIFT feature
        self.frames = {}
        self.location_3d = location

    def GetID(self):
        return self.ID

    def GetFrame(self, frame_id):
        return self.frames.get(frame_id)
    
    def SubsetOfFrames(self, frame_id):
        return {frame_id : self.frames[frame_id]}
        #subset = []
        #for frame,uv,desc in self.frames.values():
        #    if frame.GetID() == frame_id:
        #        subset.append((frame, uv, desc)) # TODO: possibly a copy needed
        #return subset
    
    def AddFrame(self, frame, uv, descriptor):
        self.frames[frame.GetID()] = (frame, uv, descriptor)


    def UpdatePoint(self, new_location):
        self.location_3d = new_location

    # checks if frame with frame_id sees this point
    def IsVisibleTo(self, frame_id):
        for frame, uv, descriptor in self.frames.values():
            if (frame_id == frame.ID):
                return True
        return False
    
    # Gets image point (2d) based on frame id
    def GetImagePoint(self, frame_id):
        ret = self.frames.get(frame_id)
        if ret != None:
            _, uv, descriptor = ret
            return (uv, descriptor)
        else: 
            return None
        #for frame, uv, descriptor in self.frames.values():
        #    if (frame_id == frame.ID):
        #        return (uv, descriptor)
        #return None
            
    def Get3dPoint(self):
        return self.location_3d
    
    def GetVectorNorm(self):
        return np.linalg.norm(self.location_3d)

    def GetNVisibleFrames(self):
        return len(self.frames)
        