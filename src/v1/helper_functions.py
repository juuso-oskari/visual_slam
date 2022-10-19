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
    return pts_l_norm, pts_r_norm

# Expects pts1 and pts2 to be matched and normalized with intrinsics
def estimateEssential(pts1, pts2, essTh):
    #E, inliers = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0/essTh) # threshold=3.0 / essTh
    E, inliers = cv2.findEssentialMat(pts1, pts2, method=cv2.LMEDS) # threshold=3.0 / essTh
    return E, inliers
        
# Expects pts1 and pts2 to be matched and normalized with intrinsics
def estimateHomography(pts1, pts2, homTh):
    #H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=3.0/homTh)
    H, inliers = cv2.findHomography(pts1, pts2, cv2.LMEDS)
    return H, inliers

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
    #print(np.shape(M1))
    c2 = np.linalg.lstsq(-M2, P2[:,3], rcond=None)[0]
    y = c2 - c1
    u1 = np.concatenate((points1, np.ones((numPoints,1))), axis=1)
    u2 = np.concatenate((points2, np.ones((numPoints,1))), axis=1)
    #u1 = [points1, ones(numPoints, 1, 'like', points1)]'
    #u2 = [points2, ones(numPoints, 1, 'like', points1)]'
    a1 = np.linalg.lstsq(M1, u1.T, rcond=None)[0]
    a2 = np.linalg.lstsq(M2, u2.T, rcond=None)[0]
    #isCodegen  = ~isempty(coder.target);
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


def estimateRelativePose(tform, inlier_pts1, inlier_pts2, K, tform_type = "Homography"):
    if tform_type == "Homography":
        # decompose homography into 4 possible solutions
        num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(tform, K)
        # choose realizable solutions according to cheirality check
        R, t, validFraction = chooseRealizableSolution(Rs, Ts, K, inlier_pts1, inlier_pts2)
        return R[0], t[0], validFraction
        
    elif tform_type == "Essential":
        # recoverpose way:
        print(np.shape(inlier_pts1))
        points, R, t, inliers = cv2.recoverPose(tform, inlier_pts1, inlier_pts2, cameraMatrix=K)
        print(R)
        validFraction = np.sum(inliers) / len(inliers) 
        # decompose essential matrix into 4 possible solutions
        #R1, R2, t = cv2.decomposeEssentialMat(tform)
        # The possible solutions are (R1,t), (R1,-t), (R2,t), (R2,-t)
        #R1, R2, t = R1[np.newaxis,:], R2[np.newaxis,:], t[np.newaxis,:]
        #Rs = np.concatenate((R1, R1, R2, R2), axis=0)
        #Ts = np.concatenate((t,-t,t,-t))
        # choose realizable solutions according to cheirality check
        #R, t, validFraction = chooseRealizableSolution(Rs, Ts, K, inlier_pts1, inlier_pts2)
        return R[0], t[0], validFraction
    else:
        print("Unknown tform_type")
        return None, None
    
def triangulation(kp1, kp2, T_1w, T_2w):
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
    return X[:3], X1, X2