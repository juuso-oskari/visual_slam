import numpy as np
import cv2
import sys

# Matches and normalizes keypoints in 2 frames, needed in many estimations to prevent numerical instability
def MatchAndNormalize(kp1, kp2, matches, K):
    # match keypoints
    pts1 = []
    pts2 = []
    for i,(m) in enumerate(matches):
        #print(m.distance)
        pts2.append(kp2[m[0].trainIdx].pt)
        pts1.append(kp1[m[0].queryIdx].pt)
    pts1  = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    # normalize points
    pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K, distCoeffs=None)
    pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K, distCoeffs=None)
    return np.squeeze(pts_l_norm), np.squeeze(pts_r_norm)

def MatchPoints(kp1, kp2, matches):
    # match keypoints
    pts1 = []
    pts2 = []
    for i,(m) in enumerate(matches):
        #print(m.distance)
        pts2.append(kp2[m[0].trainIdx].pt)
        pts1.append(kp1[m[0].queryIdx].pt)
    pts1  = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    # normalize points
    #pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K, distCoeffs=None)
    #pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K, distCoeffs=None)
    return pts1, pts2

# used in transformation score calculation
def matlab_max(v, s):
        return [max(v[i],s) for i in range(len(v))]

# Expects pts1 and pts2 to be matched and normalized with intrinsics
def estimateEssential(pts1, pts2, K, essTh):
    #E, inliers = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0/essTh) # threshold=3.0 / essTh
    pts1 = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K, distCoeffs=None)
    pts2 = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K, distCoeffs=None)
    pts1, pts2 = np.squeeze(pts1), np.squeeze(pts2)
    E, inliers = cv2.findEssentialMat(pts1, pts2,  method=cv2.RANSAC, prob=0.999, threshold=essTh) # threshold=3.0 / essTh
    # https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
    inlierPoints1 = pts1[inliers[:, 0] == 1, :]
    inlierPoints2 = pts2[inliers[:, 0] == 1, :]
    
    
    lineIn1 = cv2.computeCorrespondEpilines(inlierPoints2.reshape(-1,1,2), 2,E) # original with F
    lineIn1 = lineIn1.reshape(-1,3)
    

    inliersIndex  = np.where(inliers==1)

    locations1 = (np.concatenate(    (inlierPoints1, np.ones((np.shape(inlierPoints1)[0], 1)))    , axis=1))
    locations2 = (np.concatenate(    (inlierPoints2, np.ones((np.shape(inlierPoints2)[0], 1)))   , axis=1))
    
    error2in1 = (np.sum(locations1 * lineIn1, axis = 1))**2 / np.sum(lineIn1[:,:3]**2, axis=1)
    
    lineIn2 = cv2.computeCorrespondEpilines(inlierPoints1.reshape(-1,1,2), 2,E) # original with F
    lineIn2 = lineIn2.reshape(-1,3)
    
    error1in2 = (np.sum(locations2 * lineIn2, axis = 1))**2 / np.sum(lineIn2[:,:3]**2, axis=1)
    
    
    outlierThreshold = 4

    score = np.sum(matlab_max(outlierThreshold-error1in2, 0)) + sum(matlab_max(outlierThreshold-error2in1, 0))



    return E, inliers, score
        
# Expects pts1 and pts2 to be matched and normalized with intrinsics
def estimateHomography(pts1, pts2, homTh):
    #H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=3.0/homTh)
    H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=homTh)

    inlierPoints1 = pts1[inliers[:, 0] == 1, :]
    inlierPoints2 = pts2[inliers[:, 0] == 1, :]

    inliersIndex  = np.where(inliers==1)

    locations1 = (np.concatenate(    (inlierPoints1, np.ones((np.shape(inlierPoints1)[0], 1)))    , axis=1))
    locations2 = (np.concatenate(    (inlierPoints2, np.ones((np.shape(inlierPoints2)[0], 1)))   , axis=1))
    xy1In2     = (H @ locations1.T).T
    xy2In1     = (np.linalg.inv(H) @ locations2.T).T
    error1in2  = np.sum((locations2 - xy1In2)**2, axis=1)
    error2in1  = np.sum((locations1 - xy2In1)**2, axis=1)

    outlierThreshold = 6

    score = np.sum(matlab_max(outlierThreshold-error1in2, 0)) + np.sum(matlab_max(outlierThreshold-error2in1, 0))

    return H, inliers, score

def triangulateMidPoint(points1, points2, P1, P2):
    points1 = np.squeeze(points1)
    points2 = np.squeeze(points2)
    numPoints = np.shape(points1)[0]
    points3D = np.zeros((numPoints,3))
    P1 = P1.T
    P2 = P2.T
    M1 = P1[:3, :3]
    M2 = P2[:3, :3]
    # Get least-squares solution
    c1 = np.linalg.lstsq(-M1,  P1[:,3], rcond=None)[0]
    c2 = np.linalg.lstsq(-M2, P2[:,3], rcond=None)[0]
    y = c2 - c1
    u1 = np.concatenate((points1, np.ones((numPoints,1))), axis=1)
    u2 = np.concatenate((points2, np.ones((numPoints,1))), axis=1)
    #u1 = [points1, ones(numPoints, 1, 'like', points1)]'
    #u2 = [points2, ones(numPoints, 1, 'like', points1)]'
    a1 = np.linalg.lstsq(M1, u1.T, rcond=None)[0]
    a2 = np.linalg.lstsq(M2, u2.T, rcond=None)[0]    
    condThresh = 2**(-52)
    for i in range(numPoints):
        A   = np.array([a1[:,i], -a2[:,i]]).T 
        AtA = A.T@A
        if np.linalg.cond(AtA) < condThresh: # original: rcond(AtA) < condThresh
            # Guard against matrix being singular or ill-conditioned
            p    = np.inf(3, 1)
            p[2] = -p[2]
        else:
            alpha = np.linalg.lstsq(A, y, rcond=None)[0]
            p = (c1 + (alpha[0] * a1[:,i]).T + c2 + (alpha[1] * a2[:,i]).T) / 2
            
        points3D[i, :] = p.T

    return points3D

def chooseRealizableSolution(Rs, Ts, K, points1, points2):
    # Rs is 4x3x3, holding all possible solutions of Rotation matrix
    # Ts is 4x3x1, holding all possible solutions of Translation vector
    numNegatives = np.zeros((np.shape(Ts)[0], 1))
    #  The camera matrix is computed as follows:
    #  camMatrix = [rotationMatrix; translationVector] * K
    #  where K is the intrinsic matrix.
    #camMatrix1 = cameraMatrix(cameraParams1, np.eye(3), np.array([0 0 0]));
    camMatrix1 = np.concatenate((np.eye(3), np.zeros((1,3))), axis=0) @ K
    
    for i in range(np.shape(Ts)[0]):
        #camMatrix2 = cameraMatrix(cameraParams2, Rs(:,:,i)', Ts(i, :));
        #camMatrix2 is 4x3 @ 3x3 matmul
        camMatrix2 = np.concatenate((Rs[i].T, Ts[i].T),axis=0) @ K
        m1 = triangulateMidPoint(points1, points2, camMatrix1, camMatrix2)
        #m2 = bsxfun(@plus, m1 * Rs(:,:,i)', Ts(i, :));
        m2 = (m1 @ (Rs[i]).T) + Ts[i].T
        numNegatives[i] = np.sum((m1[:,2] < 0) | (m2[:,2] < 0))
        
    val = np.min(numNegatives)
    idx = np.where(numNegatives==val)
    validFraction = 1 - (val / points1.shape[0])
    
    R = np.zeros((len(idx), 3,3))
    t = np.zeros((len(idx), 3))
    for n in range(len(idx)):
        idx_n = idx[n][0]
        R0 = Rs[idx_n].T
        t0 = Ts[idx_n].T

        tNorm = np.linalg.norm(t0)
        if tNorm != 0:
            t0 = t0 / tNorm
        R[n] = R0
        t[n] = t0

    return R, t, validFraction


def estimateRelativePose(tform, inlier_pts1, inlier_pts2, K, tform_type = "Essential"):
    if tform_type == "Homography":
        # decompose homography into 4 possible solutions
        num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(tform, K=np.eye(3))
        # choose realizable solutions according to cheirality check
        R, t, validFraction = chooseRealizableSolution(Rs, Ts, K, inlier_pts1, inlier_pts2)
        if np.shape(R)[0] >= 2: # TODO: better way to choose from 2 realizable
                R,t = R[1], t[1]
        return R, t, validFraction
        
    elif tform_type == "Essential":
        # recoverpose way:
        points, R, t, inliers = cv2.recoverPose(tform, inlier_pts1, inlier_pts2, cameraMatrix=K)
        validFraction = points / np.shape(inliers)[0]
        return R, t, validFraction 
        # decompose essential matrix into 4 possible solutions
        R1, R2, t = cv2.decomposeEssentialMat(tform)
        # The possible solutions are (R1,t), (R1,-t), (R2,t), (R2,-t)
        R1, R2, t = R1[np.newaxis,:], R2[np.newaxis,:], t[np.newaxis,:]
        Rs = np.concatenate((R1, R1, R2, R2), axis=0)
        Ts = np.concatenate((t,-t,t,-t))
        # choose realizable solutions according to cheirality check
        R, t, validFraction = chooseRealizableSolution(Rs, Ts, K, inlier_pts1, inlier_pts2)
        if np.shape(R)[0] >= 2:
                R,t = R[1], t[1]
        return R, t, validFraction
    else:
        print("Unknown tform_type")
        return None, None, 0
    
def triangulation(kp1, kp2, T_1w, T_2w, reprojection_threshold = 1, min_parallax = 4):
    """Triangulation to get 3D points
    Initial version of trigualation
    Might be error prone
    Args:
        kp1 (Nx2): inlier keypoint in view 1 (normalized)
        kp2 (Nx2): inlier keypoints in view 2 (normalized)
        T_1w (4x4): pose of view 1 w.r.t  i.e. T_1w (from w to 1)
        T_2w (4x4): pose of view 2 w.r.t world, i.e. T_2w (from w to 2)
    Returns:
        X (3xN): 3D coordinates of the keypoints w.r.t world coordinate
        X1 (3xN): 3D coordinates of the keypoints w.r.t view1 coordinate
        X2 (3xN): 3D coordinates of the keypoints w.r.t view2 coordinate
    """
    kp1_3D = np.ones((3, kp1.shape[0]))
    kp2_3D = np.ones((3, kp2.shape[0]))
    kp1_3D[0], kp1_3D[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
    kp2_3D[0], kp2_3D[1] = kp2[:, 0].copy(), kp2[:, 1].copy()
    X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
    X /= X[3]
    X1 = T_1w[:3] @ X
    X2 = T_2w[:3] @ X
    # get reprojection error
    # project world points (in camera 2 reference frame) back to image plane (image plane of camera 2)
    # Our poses (estimated from essential matrix) already account for camera intrinsics (K)
    proj_points2 = X2
    proj_points2 /= proj_points2[2] # normalize to homogenous coordinates
    err2 = np.abs(kp2 - proj_points2[:2].T)
    # do the same for first image
    proj_points1 = X1
    proj_points1 /= proj_points1[2]
    err1 = np.abs(kp1 - proj_points1[:2].T)
    reprojection_error = np.mean(np.concatenate((err1, err2), axis=0), axis=1)
    # a good two-view with significant parallax
    ray1 = X - T_1w.t
    ray2 = X - T_2w.t
    cosangle = np.sum(ray1 * ray2, axis=0) / (   np.linalg.norm(ray1, axis=0)*np.linalg.norm(ray2, axis=0)  )    
    
    # get inliers
    inliers = reprojection_error < reprojection_threshold and cosangle > np.arccos(min_parallax)
    
    return X[:3], X1, X2, inliers