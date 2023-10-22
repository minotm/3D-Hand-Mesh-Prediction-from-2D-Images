import numpy as np
import torch
import cv2


def get_num_images(split, num_images):
    if split in ["train", "val", "test"]:
        return num_images

    if split == "smalltrain":
        return 100000

    if split == "tinytrain":
        return 12000

    if split == "minitrain":
        return 300

    if split == "smallval":
        return 12000

    if split == "tinyval":
        return 6000

    if split == "minival":
        return 200

    if split == "smalltest":
        return 12000

    if split == "tinytest":
        return 6000

    if split == "minitest":
        return 200

    assert False, f"Invalid split {split}"


def subtract_root(joints: np.ndarray, root_idx: int):
    joints_ra = joints.copy()
    root = joints_ra[root_idx : root_idx + 1]
    joints_ra -= root
    return joints_ra


def subtract_root_batch(joints: torch.Tensor, root_idx: int):
    assert len(joints.shape) == 3
    assert joints.shape[2] == 3
    joints_ra = joints.clone()
    root = joints_ra[:, root_idx : root_idx + 1]
    joints_ra -= root
    return joints_ra


def pad_joints(joints, root_idx):
    num_joints = joints.shape[0]
    num_dim = joints.shape[1]
    joints_conf = np.ones((num_joints, num_dim + 1))
    if root_idx is None:
        joints_conf[:, :num_dim] = joints
    else:
        joints_conf[:, :num_dim] = (
            joints - joints[root_idx : root_idx + 1]
        )  # root relative
    return joints_conf


def downsample(fnames, split):
    if "small" not in split and "mini" not in split:
        return fnames
    import random

    random.seed(1)
    assert (
        random.randint(0, 100) == 17
    ), "Same seed but different results; Subsampling might be different."
    all_keys = fnames
    if split == "minival" or split == "minitest":
        num_samples = 1000
    elif split == "smallval" or split == "smalltest":
        num_samples = int(0.25 * len(all_keys))
    elif split == "minitrain":
        num_samples = 1000
    elif split == "smalltrain":
        num_samples = int(0.1 * len(all_keys))
    elif split == "minivis":
        num_samples = 32
    else:
        assert False, "Unknown split {}".format(split)
    curr_keys = random.sample(all_keys, num_samples)
    return curr_keys


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(
    c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False
):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def generate_patch_image(
    cvimg,
    bbox,
    scale,
    rot,
    out_shape,
    interpl_strategy,
    gauss_kernel=5,
    gauss_sigma=8.0,
):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0])
    bb_c_y = float(bbox[1])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = gen_trans_from_patch_cv(
        bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot
    )

    # anti-aliasing
    blur = cv2.GaussianBlur(img, (gauss_kernel, gauss_kernel), gauss_sigma)
    img_patch = cv2.warpAffine(
        blur, trans, (int(out_shape[1]), int(out_shape[0])), flags=interpl_strategy
    )
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(
        bb_c_x,
        bb_c_y,
        bb_width,
        bb_height,
        out_shape[1],
        out_shape[0],
        scale,
        rot,
        inv=True,
    )

    return img_patch, trans, inv_trans
