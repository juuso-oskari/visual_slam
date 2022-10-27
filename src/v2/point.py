import numpy as np
import cv2

class Point:
    def __init__(self, location, id):
        self.ID = id
        self.frames = [] # list of tuples. (frame, (u,v)). Where u,v is the 2d location on the image plane of the frame
        self.location_3d = location

    def AddFrame(self, frame,uv):
        u = uv[0]
        v = uv[1]
        self.frames.append((frame, (u,v)))


    def UpdatePoint(self, new_location):
        self.location_3d = new_location

    # Gets image point (2d) based on frame id
    def GetImagePoint(self, frame_id):
        for frame, uv in self.frames:
            if (frame_id == frame.ID):
                return uv