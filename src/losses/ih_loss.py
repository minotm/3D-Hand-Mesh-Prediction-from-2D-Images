import torch
import torch.nn as nn
from core.py3d import axis_angle_to_matrix
from src.datasets.dataset_utils import subtract_root_batch


class IHLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, pred, gt):
        # unpacking pred and gt
        pred_betas_r = pred["mano.beta.r"]
        pred_rotmat_r = pred["mano.pose.r"]
        pred_joints_r = pred["mano.joints3d.cam.patch.r"]
        pred_projected_keypoints_2d_r = pred["mano.joints2d.norm.r"]
        pred_betas_l = pred["mano.beta.l"]
        pred_rotmat_l = pred["mano.pose.l"]
        pred_joints_l = pred["mano.joints3d.cam.patch.l"]
        pred_projected_keypoints_2d_l = pred["mano.joints2d.norm.l"]

        gt_pose_r = gt["mano.pose.r"]
        gt_betas_r = gt["mano.beta.r"]
        gt_joints_r = gt["mano.joints3d.cam.patch.r"]
        gt_keypoints_2d_r = gt["mano.joints2d.norm.r"]
        gt_pose_l = gt["mano.pose.l"]
        gt_betas_l = gt["mano.beta.l"]
        gt_joints_l = gt["mano.joints3d.cam.patch.l"]
        gt_keypoints_2d_l = gt["mano.joints2d.norm.l"]

        right_valid = gt["right_valid"]
        left_valid = gt["left_valid"]

        # axis angle to rot mat
        gt_pose_r = axis_angle_to_matrix(gt_pose_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
        gt_pose_l = axis_angle_to_matrix(gt_pose_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)

        # Compute loss on MANO parameters
        loss_regr_pose_r, loss_regr_betas_r = mano_loss(
            pred_rotmat_r,
            pred_betas_r,
            gt_pose_r,
            gt_betas_r,
            criterion=self.mse_loss,
            is_valid=right_valid,
        )
        loss_regr_pose_l, loss_regr_betas_l = mano_loss(
            pred_rotmat_l,
            pred_betas_l,
            gt_pose_l,
            gt_betas_l,
            criterion=self.mse_loss,
            is_valid=left_valid,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints_r = mask_loss(
            pred_projected_keypoints_2d_r,
            gt_keypoints_2d_r,
            criterion=self.mse_loss,
            is_valid=right_valid,
        )
        loss_keypoints_l = mask_loss(
            pred_projected_keypoints_2d_l,
            gt_keypoints_2d_l,
            criterion=self.mse_loss,
            is_valid=left_valid,
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d_r = hand_kp3d_loss(
            pred_joints_r, gt_joints_r, self.mse_loss, right_valid
        )
        loss_keypoints_3d_l = hand_kp3d_loss(
            pred_joints_l, gt_joints_l, self.mse_loss, left_valid
        )

        # relative translation between two hands
        both_valid = right_valid * left_valid
        flag = ".wp"
        diff_pred = pred[f"mano.cam_t{flag}.l"] - pred[f"mano.cam_t{flag}.r"]
        diff_gt = gt[f"mano.cam_t{flag}.l"] - gt[f"mano.cam_t{flag}.r"]
        loss_transl = mask_loss(diff_pred, diff_gt, self.mse_loss, both_valid)

        # reweighting
        lambda_mano_pose = 10 #updated 19.05 mm
        lambda_beta = 1.0
        lambda_kp2d = 10.0
        lambda_kp3d = 100.0 #updated 19.05 mm
        lambda_transl = 5.0

        loss_regr_pose_r *= lambda_mano_pose
        loss_regr_pose_l *= lambda_mano_pose
        loss_regr_betas_r *= lambda_beta
        loss_regr_betas_l *= lambda_beta

        loss_keypoints_r *= lambda_kp2d
        loss_keypoints_l *= lambda_kp2d
        loss_keypoints_3d_r *= lambda_kp3d
        loss_keypoints_3d_l *= lambda_kp3d
        loss_transl *= lambda_transl

        loss_dict = {
            "loss/mano/kp2d/r": loss_keypoints_r,
            "loss/mano/kp3d/r": loss_keypoints_3d_r,
            "loss/mano/pose/r": loss_regr_pose_r,
            "loss/mano/beta/r": loss_regr_betas_r,
            "loss/mano/kp2d/l": loss_keypoints_l,
            "loss/mano/kp3d/l": loss_keypoints_3d_l,
            "loss/mano/pose/l": loss_regr_pose_l,
            "loss/mano/transl": loss_transl,
            "loss/mano/beta/l": loss_regr_betas_l,
        }
        return loss_dict


def mask_loss(pred_vector, gt_vector, criterion, is_valid=None):
    if is_valid is not None and is_valid.long().sum() > 0:
        valid_idx = is_valid.long().bool()
        dist = criterion(pred_vector[valid_idx], gt_vector[valid_idx])
        loss = dist.mean().view(-1)
    else:
        loss = torch.FloatTensor([0.0]).to(gt_vector.device)
    return loss


def mano_loss(pred_rotmat, pred_betas, gt_rotmat, gt_betas, criterion, is_valid=None):
    loss_regr_pose = mask_loss(pred_rotmat, gt_rotmat, criterion, is_valid)
    loss_regr_betas = mask_loss(pred_betas, gt_betas, criterion, is_valid)
    return loss_regr_pose, loss_regr_betas


def object_kp3d_loss(pred_3d, gt_3d, criterion, is_valid):
    num_kps = pred_3d.shape[1] // 2
    pred_3d_ra = subtract_root_batch(pred_3d, root_idx=num_kps)
    gt_3d_ra = subtract_root_batch(gt_3d, root_idx=num_kps)
    loss_kp = keypoint_3d_loss(
        pred_3d_ra,
        gt_3d_ra,
        criterion=criterion,
        is_valid=is_valid,
    )
    return loss_kp


def hand_kp3d_loss(pred_3d, gt_3d, criterion, is_valid):
    pred_3d_ra = subtract_root_batch(pred_3d, root_idx=0)
    gt_3d_ra = subtract_root_batch(gt_3d, root_idx=0)
    loss_kp = keypoint_3d_loss(
        pred_3d_ra,
        gt_3d_ra,
        criterion=criterion,
        is_valid=is_valid,
    )
    return loss_kp


def keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, criterion, is_valid=None):
    """
    Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    gt_root = gt_keypoints_3d[:, :1, :]
    gt_keypoints_3d = gt_keypoints_3d - gt_root
    pred_root = pred_keypoints_3d[:, :1, :]
    pred_keypoints_3d = pred_keypoints_3d - pred_root
    return mask_loss(pred_keypoints_3d, gt_keypoints_3d, criterion, is_valid)
