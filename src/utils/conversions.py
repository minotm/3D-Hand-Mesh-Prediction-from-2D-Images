import torch
import numpy as np
from src.utils.geometry import perspective_projection


def kp3d_to_kp2d(cam_t, pts3d, image_size):
    """
    This function project 3d keypoints to 2d in the unormlized space.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        camera_center (bs, 2): Camera center
    """

    assert isinstance(cam_t, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert len(cam_t.shape) == 2
    bs = cam_t.shape[0]
    assert pts3d.shape[0] == bs
    assert pts3d.shape[2] == 3
    assert len(image_size) == 2

    rotation = torch.eye(3)[None].repeat(bs, 1, 1).to(pts3d.device)
    camera_centers = torch.FloatTensor(image_size)[None].repeat(bs, 1) / 2.0
    xy2d = perspective_projection(
        pts3d, rotation, cam_t, 1000.0, camera_centers.to(pts3d.device)
    )
    return xy2d


def normalize_kp2d_np(kp2d: np.ndarray, img_res):
    assert kp2d.shape[1] == 3
    kp2d_normalized = kp2d.copy()
    kp2d_normalized[:, :2] = 2.0 * kp2d[:, :2] / img_res - 1.0
    return kp2d_normalized


def unnormalize_2d_kp(kp_2d_np: np.ndarray, res):
    assert kp_2d_np.shape[1] == 3
    kp_2d = np.copy(kp_2d_np)
    kp_2d[:, :2] = 0.5 * res * (kp_2d[:, :2] + 1)
    return kp_2d


def normalize_kp2d(kp2d: torch.Tensor, img_res):
    assert len(kp2d.shape) == 3
    kp2d_normalized = kp2d.clone()
    kp2d_normalized[:, :, :2] = 2.0 * kp2d[:, :, :2] / img_res - 1.0
    return kp2d_normalized


def unormalize_kp2d(kp2d_normalized: torch.Tensor, img_res):
    assert len(kp2d_normalized.shape) == 3
    assert kp2d_normalized.shape[2] == 2
    kp2d = kp2d_normalized.clone()
    kp2d = 0.5 * img_res * (kp2d + 1)
    return kp2d
