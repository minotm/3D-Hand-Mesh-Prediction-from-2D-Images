import torch
import numpy as np
from torch.nn import functional as F

"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""


def weak_perspective_to_perspective_torch(
    weak_perspective_camera, focal_length, img_res
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    s = weak_perspective_camera[:, 0]
    tx = weak_perspective_camera[:, 1]
    ty = weak_perspective_camera[:, 2]
    perspective_camera = torch.stack(
        [
            tx,
            ty,
            2 * focal_length / (img_res * s + 1e-9),
        ],
        dim=-1,
    )
    return perspective_camera


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(batch_size, 3, 3)
    return rotMat


def batch_aa2rot(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    assert len(axisang.shape) == 2
    assert axisang.shape[1] == 3
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum("bij,bkj->bki", rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)

    return projected_points[:, :, :-1]


def estimate_translation_k(
    S,
    joints_2d,
    K,
    use_all_joints=False,
    rotation=None,
    pad_2d=False,
):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """
    if pad_2d:
        batch, num_pts = joints_2d.shape[:2]
        joints_2d_pad = torch.ones((batch, num_pts, 3))
        joints_2d_pad[:, :, :2] = joints_2d
        joints_2d_pad = joints_2d_pad.to(joints_2d.device)
        joints_2d = joints_2d_pad

    device = S.device

    if rotation is not None:
        S = torch.einsum("bij,bkj->bki", rotation, S)

    # Use only joints 25:49 (GT joints)
    if use_all_joints:
        S = S.cpu().numpy()
        joints_2d = joints_2d.cpu().numpy()
    else:
        S = S[:, 25:, :].cpu().numpy()
        joints_2d = joints_2d[:, 25:, :].cpu().numpy()

    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        K_i = K[i]
        if np.isnan(S_i.sum()):
            trans[i] = float("nan")
        else:
            trans[i] = estimate_translation_k_np(S_i, joints_i, conf_i, K_i)
    return torch.from_numpy(trans).to(device)


def estimate_translation_k_np(S, joints_2d, joints_conf, K):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """
    num_joints = S.shape[0]
    # focal length

    focal = np.array([K[0, 0], K[1, 1]])
    # optical center
    center = np.array([K[0, 2], K[1, 2]])

    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(focal, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array(
        [
            F * np.tile(np.array([1, 0]), num_joints),
            F * np.tile(np.array([0, 1]), num_joints),
            O - np.reshape(joints_2d, -1),
        ]
    ).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def solve_rigid_tf_np(A: np.ndarray, B: np.ndarray):
    """
    “Least-Squares Fitting of Two 3-D Point Sets”, Arun, K. S. , May 1987
    Input: expects Nx3 matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector

    This function should be a fix for compute_rigid_tf when the det == -1
    """

    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def batch_solve_rigid_tf(A, B):
    """
    “Least-Squares Fitting of Two 3-D Point Sets”, Arun, K. S. , May 1987
    Input: expects BxNx3 matrix of points
    Returns R,t
    R = Bx3x3 rotation matrix
    t = Bx3x1 column vector
    """

    assert A.shape == B.shape
    A = A.permute(0, 2, 1)
    B = B.permute(0, 2, 1)

    batch, num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    _, num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = torch.mean(A, dim=2)
    centroid_B = torch.mean(B, dim=2)

    # ensure centroids are 3x1
    centroid_A = centroid_A.view(batch, -1, 1)
    centroid_B = centroid_B.view(batch, -1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = torch.bmm(Am, Bm.permute(0, 2, 1))

    # find rotation
    U, S, Vt = torch.svd(H)
    Vt = Vt.permute(0, 2, 1)
    R = torch.bmm(Vt.permute(0, 2, 1), U.permute(0, 2, 1))

    # special reflection case
    neg_idx = torch.det(R) < 0
    if neg_idx.sum() > 0:
        raise Exception(
            f"some rotation matrices are not orthogonal; make sure implementation is correct for such case: {neg_idx}"
        )
    Vt[neg_idx, 2, :] *= -1
    R[neg_idx, :, :] = torch.bmm(
        Vt[neg_idx].permute(0, 2, 1), U[neg_idx].permute(0, 2, 1)
    )

    t = torch.bmm(-R, centroid_A) + centroid_B
    return R, t


def rigid_tf_torch_batch(points, R, T):
    """
    Performs rigid transformation to incoming points but batched
    Q = (points*R.T) + T
    points: (batch, num, 3)
    R: (batch, 3, 3)
    T: (batch, 3, 1)
    out: (batch, num, 3)
    """
    points_out = torch.bmm(R, points.permute(0, 2, 1)) + T
    points_out = points_out.permute(0, 2, 1)
    return points_out


def weak_perspective_to_perspective_torch(
    weak_perspective_camera, focal_length, img_res
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    s = weak_perspective_camera[:, 0]
    tx = weak_perspective_camera[:, 1]
    ty = weak_perspective_camera[:, 2]
    perspective_camera = torch.stack(
        [
            tx,
            ty,
            2 * focal_length / (img_res * s + 1e-9),
        ],
        dim=-1,
    )
    return perspective_camera


def perspective_to_weak_perspective_torch(
    perspective_camera,
    focal_length,
    img_res,
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    # if isinstance(focal_length, torch.Tensor):
    #     focal_length = focal_length[:, 0]

    tx = perspective_camera[:, 0]
    ty = perspective_camera[:, 1]
    tz = perspective_camera[:, 2]

    weak_perspective_camera = torch.stack(
        [2 * focal_length / (img_res * tz + 1e-9), tx, ty],
        dim=-1,
    )
    return weak_perspective_camera
