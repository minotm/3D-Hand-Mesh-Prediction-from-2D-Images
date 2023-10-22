import torch
import numpy as np


def compute_mrrpe(root_r_gt, root_l_gt, root_r_pred, root_l_pred, is_valid):
    rel_vec_gt = root_l_gt - root_r_gt
    rel_vec_pred = root_l_pred - root_r_pred

    invalid_idx = (1 - is_valid).long().nonzero().view(-1)
    mrrpe = ((rel_vec_pred - rel_vec_gt) ** 2).sum(dim=1).sqrt()
    mrrpe[invalid_idx] = float("nan")
    mrrpe = mrrpe.cpu().numpy()
    return mrrpe


def compute_joint3d_error(joints3d_cam_gt, joints3d_cam_pred, valid_jts):
    dist = ((joints3d_cam_gt - joints3d_cam_pred) ** 2).sum(dim=2).sqrt()
    invalid_idx = (1 - valid_jts).long().nonzero().view(-1)
    dist[invalid_idx, :] = float("nan")
    dist = dist.cpu().numpy()
    return dist


def evaluate_metrics(pred, targets, img_res):
    # unpacking
    joints3d_cam_r_gt = targets["mano.joints3d.cam.patch.r"]
    joints3d_cam_l_gt = targets["mano.joints3d.cam.patch.l"]

    left_valid = targets["left_valid"]
    right_valid = targets["right_valid"]

    joints3d_cam_r_pred = pred["mano.joints3d.cam.patch.r"]
    joints3d_cam_l_pred = pred["mano.joints3d.cam.patch.l"]

    # compute relative metrics
    joints3d_cam_r_gt_ra = joints3d_cam_r_gt - joints3d_cam_r_gt[:, :1, :]
    joints3d_cam_l_gt_ra = joints3d_cam_l_gt - joints3d_cam_l_gt[:, :1, :]
    joints3d_cam_r_pred_ra = joints3d_cam_r_pred - joints3d_cam_r_pred[:, :1, :]
    joints3d_cam_l_pred_ra = joints3d_cam_l_pred - joints3d_cam_l_pred[:, :1, :]

    mpjpe_ra_r = compute_joint3d_error(
        joints3d_cam_r_gt_ra, joints3d_cam_r_pred_ra, right_valid
    )
    mpjpe_ra_l = compute_joint3d_error(
        joints3d_cam_l_gt_ra, joints3d_cam_l_pred_ra, left_valid
    )

    # interaction metrics
    root_r_gt = joints3d_cam_r_gt[:, 0]
    root_l_gt = joints3d_cam_l_gt[:, 0]
    root_r_pred = joints3d_cam_r_pred[:, 0]
    root_l_pred = joints3d_cam_l_pred[:, 0]
    mrrpe_rl = compute_mrrpe(
        root_r_gt, root_l_gt, root_r_pred, root_l_pred, left_valid * right_valid
    )

    # aggregate
    # hands
    metric_dict = {}
    metric_dict["mpjpe/ra/r"] = mpjpe_ra_r.mean(axis=1) * 1000
    metric_dict["mpjpe/ra/l"] = mpjpe_ra_l.mean(axis=1) * 1000
    metric_dict["mrrpe/r/l"] = mrrpe_rl * 1000
    return metric_dict
