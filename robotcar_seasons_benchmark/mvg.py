# Generic multi-view geometry functions

import numpy as np


class CameraK():
    def __init__(self, fx, fy, cx, cy):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)


class Pose:
    # Class to store pose information
    # Pose is stored in the following convention: P = R @ (Pw - t), where Pw are point coordinates in the world
    # reference frame and P are point coordinates in the camera reference frame
    def __init__(self, R, t):
        assert R.shape == (3, 3)
        assert t.shape == (3, ) or t.shape == (3, 1)

        self.R = R
        if t.shape == (3, ):
            # Reshape into (3, 1) vector
            self.t = np.reshape(t, (-1, 1))
        else:
            self.t = t


def q2r(q):
    # Rotation matrix from Hamiltonian quaternion
    # Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    w, x, y, z = tuple(q)

    n = 1.0/np.sqrt(x*x+y*y+z*z+w*w)
    x *= n
    y *= n
    z *= n
    w *= n
    r = np.array([[1.0 - 2.0*y*y - 2.0*z*z, 2.0*x*y - 2.0*z*w, 2.0*x*z + 2.0*y*w],
                  [2.0*x*y + 2.0*z*w, 1.0 - 2.0*x*x - 2.0*z*z, 2.0*y*z - 2.0*x*w],
                  [2.0*x*z - 2.0*y*w, 2.0*y*z + 2.0*x*w, 1.0 - 2.0*x*x - 2.0*y*y]])
    return r


def se3_to_rt(se3):
    # se3 pose is 4x 4 matrix, such that
    # Pw = [R | T] @ [P]
    #      [0 | 1]   [1]
    # where Pw are coordinates in the world reference frame and P are coordinates in the camera frame
    # Transforming this equation we get: Pw = R @ P + T   ==>   P = R^{-1} @ (Pw - T)
    assert se3.shape == (4, 4)
    # See above: to convert from se4 to pose in (R, t) form, we need to inverse the rotation matrix
    r = se3[0:3, 0:3].transpose()
    t = se3[0:3, 3]
    return r, t


def rt_to_se3(r, t):
    # Converts 3x3 rotation matrix and 3x1 translation vector to 4x4 SE3 transformation matrix
    assert r.shape == (3, 3)
    pose = np.eye(4, dtype=np.float64)
    pose[0:3, 0:3] = r.transpose()
    pose[0:3, 3] = t
    return pose


def p2hn(p, K):
    # Convert from pixel to homogeneous normalized coordinates
    # p are pixel coordinates of the point (u, v)
    # x_tilde = (u-cx)/fx
    # y_tilde = (v-cy)/fy
    assert K.shape == (3, 3)
    nh = np.vstack(((p[0] - K[0, 2]) / K[0, 0],
                    (p[1] - K[1, 2]) / K[1, 1], 1.0))
    return nh


def hn2p(p, K):
    # Convert from homogeneous normalized coordinates to pixel coordinates
    # p are homogeneous normalized coordinates of the point d * (x_tilde, y_tilde, 1.0)
    # u = x_tilde*fx + cx
    # v = y_tilde*fy + cy
    assert p.shape[0] == 3
    assert K.shape == (3, 3)
    p = h2e(p)
    p = np.vstack((p[0] * K[0, 0] + K[0, 2],
                  p[1] * K[1, 1] + K[1, 2]))
    return p


def e2h(p):
    # Convert from euclidean to homogeneous coordinates
    # p are 2D or 3D coordinates
    assert p.shape[0] in [2, 3]
    if p.shape[0] == 2:
        ph = np.vstack((p[0], p[1], 1.0))
    elif p.shape[0] == 3:
        ph = np.vstack((p[0], p[1], p[2], 1.0))
    return ph


def h2e(p):
    # Convert from homogeneous to euclidean coordinates
    # p are 4D (homogeneous for 3D point) or 3D (homogeneous for 2D point)
    assert p.shape[0] in [3, 4]
    epsilon = 1e-7
    assert np.abs(p[-1]) > epsilon

    if p.shape[0] == 3:
        ph = np.vstack((p[0]/p[2], p[1]/p[2]))
    elif p.shape[0] == 4:
        ph = np.vstack((p[0]/p[3], p[1]/p[3], p[2]/p[3]))
    return ph


def relative_camera_pose(pose1, pose2):
    # Returns relative pose of the second camera with respect to the first camera
    # such that below equation holds
    # P2 = R*(P1-T)
    # P1 coordinates in the first camera reference frame
    # P2 coordinates in the second camera reference frame

    assert isinstance(pose1, Pose)
    assert isinstance(pose2, Pose)

    R = pose2.R @ pose1.R.transpose()
    t = pose1.R @ (pose2.t-pose1.t)
    return Pose(R, t)


