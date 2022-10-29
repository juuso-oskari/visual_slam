# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/HelloPangolin

import OpenGL.GL as gl
import pangolin

import numpy as np



def main():
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)

    
    trajectory = [[0, -6, 6]]
    for i in range(300):
        trajectory.append(trajectory[-1] + np.random.random(3)-0.5)
    trajectory = np.array(trajectory)

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)
        """
        # Render OpenGL Cube
        pangolin.glDrawColouredCube(0.1)

        # Draw Point Cloud
        points = np.random.random((10000, 3)) * 3 - 4
        gl.glPointSize(1)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(points)
        
        # Draw Point Cloud
        points = np.random.random((10000, 3))
        colors = np.zeros((len(points), 3))
        colors[:, 1] = 1 -points[:, 0]
        colors[:, 2] = 1 - points[:, 1]
        colors[:, 0] = 1 - points[:, 2]
        points = points * 3 + 1
        gl.glPointSize(1)
        pangolin.DrawPoints(points, colors)
        """    
        # Draw lines
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 0.0)
        pangolin.DrawPoints(trajectory)   # consecutive
        # Draw camera
        pose = np.identity(4)
        pose[:3, 3] = np.random.randn(3)
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawCamera(pose, 0.03)
        """
        # Draw boxes
        poses = [np.identity(4) for i in range(10)]
        for pose in poses:
            pose[:3, 3] = np.random.randn(3) + np.array([5,-3,0])
        sizes = np.random.random((len(poses), 3))
        gl.glLineWidth(1)
        gl.glColor3f(1.0, 0.0, 1.0)
        pangolin.DrawBoxes(poses, sizes)

        """
        pangolin.FinishFrame()



if __name__ == '__main__':
    main()