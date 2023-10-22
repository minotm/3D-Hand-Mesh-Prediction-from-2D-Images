import torch
import torch.nn as nn

from src.utils.geometry import estimate_translation_k
from src.utils.geometry import batch_solve_rigid_tf, rigid_tf_torch_batch
from src.utils.rend import Renderer
import core.ld_utils as ld_utils
import src.utils.eval_utils_ih as eval_utils
import src.pl.pl_factory as pl_factory
import src.utils.conversions as conversions
from src.utils.conversions import unormalize_kp2d
from core.unidict import unidict
import src.utils.geometry as geometry
from src.utils.mano import mano_layer


class IHWrapper_no_gpu(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # modules
        self.model = pl_factory.fetch_pytorch_model(args)
        self.loss_fn = pl_factory.fetch_loss_fn(args)

        self.mano_r = mano_layer["right"]
        self.mano_l = mano_layer["left"]
        self.add_module("mano_r", self.mano_r)
        self.add_module("mano_l", self.mano_l)
        #self.renderer = Renderer(img_res=args.img_res)

    def load_pretrained(self, path):
        if path == "":
            return
        sd = torch.load(path)["network"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.load_state_dict(sd)
        print("Loaded pretrained weights: {}".format(path))

    def set_flags(self, mode):
        self.model.mode = mode
        if mode == "train":
            self.train()
        else:
            self.eval()

    def forward(self, inputs, targets, meta_info, mode):
        self.set_flags(mode)

        # python dict to a custom dictionary
        inputs = unidict(inputs)
        targets = unidict(targets)
        meta_info = unidict(meta_info)

        pred = self.model(
            {
                "inputs.images": inputs["img"],
                "meta.intrinsics": meta_info["intrinsics"],
            }
        )
        if mode == "inference":
            pred.register("imgname", meta_info["imgname"])
            return pred

        with torch.no_grad():
            targets = self.prepare_targets(targets, meta_info, mode)
        loss_dict = self.loss_fn(pred=pred, gt=targets)
        loss_dict = {k: loss_dict[k].mean() for k in loss_dict}
        loss_dict["loss"] = sum(loss_dict[k] for k in loss_dict)

        # conversion for vis and eval
        keys = list(pred.keys())
        for key in keys:
            # denormalize 2d keypoints
            if "2d.norm" in key:
                denorm_key = key.replace(".norm", "")
                assert key in targets.keys(), f"Do not have key {key}"

                val_pred = pred[key]
                val_gt = targets[key]

                val_denorm_pred = unormalize_kp2d(val_pred, self.args.img_res)
                val_denorm_gt = unormalize_kp2d(val_gt, self.args.img_res)

                pred.register(denorm_key, val_denorm_pred)
                targets.register(denorm_key, val_denorm_gt)

        if mode == "train":
            return loss_dict

        if mode == "vis":
            vis_dict = unidict()
            vis_dict.merge(inputs.prefix("inputs."))
            vis_dict.merge(pred.prefix("pred."))
            vis_dict.merge(targets.prefix("targets."))
            vis_dict.merge(meta_info.prefix("meta_info."))
            vis_dict.register("meta_info.mano.faces.r", self.mano_r.faces)
            vis_dict.register("meta_info.mano.faces.l", self.mano_l.faces)
            vis_dict = vis_dict.detach()
            return vis_dict

        # evaluate metrics
        metrics_all = unidict()
        metrics = eval_utils.evaluate_metrics(pred, targets, self.args.img_res)
        metrics_all.merge(unidict(metrics))

        for k, v in metrics_all.items():
            metrics_all.overwrite(k, torch.FloatTensor(v))

        out_dict = unidict(
            {
                "sample_index": meta_info["sample_index"],
            }
        )
        out_dict.merge(ld_utils.prefix_dict(metrics_all, "metric."))
        return out_dict, loss_dict

    def prepare_targets(self, targets, meta_info, mode):
        # unpacking
        K = meta_info["intrinsics"]
        gt_pose_r = targets["mano.pose.r"]  # MANO pose parameters
        gt_betas_r = targets["mano.beta.r"]  # MANO beta parameters

        gt_pose_l = targets["mano.pose.l"]  # MANO pose parameters
        gt_betas_l = targets["mano.beta.l"]  # MANO beta parameters

        # normalized 2d keypoints
        gt_joints2d_norm_r = targets["mano.joints2d.norm.r"]
        gt_joints2d_norm_l = targets["mano.joints2d.norm.l"]

        gt_joints2d_r = conversions.unormalize_kp2d(
            gt_joints2d_norm_r, self.args.img_res
        )

        gt_joints2d_l = conversions.unormalize_kp2d(
            gt_joints2d_norm_l, self.args.img_res
        )

        # 3D joints in camera coordinate
        # the joints are in the camera coordinate of the full image
        # not the camera coordinate of the crop
        gt_joints3d_r_full = targets["mano.joints3d.cam.full.r"]
        gt_joints3d_l_full = targets["mano.joints3d.cam.full.l"]

        left_valid = targets["left_valid"]
        right_valid = targets["right_valid"]
        both_valid = left_valid * right_valid
        valid_idx = both_valid.bool()

        # MANO without translation
        gt_out_r = self.mano_r(
            betas=gt_betas_r, hand_pose=gt_pose_r[:, 3:], global_orient=gt_pose_r[:, :3]
        )
        gt_model_joints_r = gt_out_r.joints
        gt_vertices_r = gt_out_r.vertices

        gt_out_l = self.mano_l(
            betas=gt_betas_l, hand_pose=gt_pose_l[:, 3:], global_orient=gt_pose_l[:, :3]
        )
        gt_model_joints_l = gt_out_l.joints
        gt_vertices_l = gt_out_l.vertices
        gt_root_cano_l = gt_out_l.joints[:, 0]

        # solve for camera translation for single hand
        gt_cam_t_r = estimate_translation_k(
            gt_model_joints_r,
            gt_joints2d_r,
            meta_info["intrinsics"].cpu().numpy(),
            use_all_joints=True,
            pad_2d=True,
        )

        # separatley for the left hand
        gt_cam_t_l = estimate_translation_k(
            gt_model_joints_l,
            gt_joints2d_l,
            meta_info["intrinsics"].cpu().numpy(),
            use_all_joints=True,
            pad_2d=True,
        )

        # move to camera coord of the patch, from the coord of the full image
        if mode != "train":
            gt_vertices_r = gt_vertices_r + gt_cam_t_r[:, None, :]
        gt_model_joints_r = gt_model_joints_r + gt_cam_t_r[:, None, :]

        # for interacting hand images
        # transform the relative translation of the left hand root to the right hand
        # from the original full image coordinate to the patch coordinate
        if valid_idx.sum() > 0:
            gt_joints3d_r_valid = gt_joints3d_r_full[valid_idx]
            gt_joints3d_l_valid = gt_joints3d_l_full[valid_idx]

            # right hand is in camera coord (for the patch)
            # map left hand to it while preserving rel. dist.
            gt_model_joints_r_valid = gt_model_joints_r[valid_idx]
            R, t = batch_solve_rigid_tf(gt_joints3d_r_valid, gt_model_joints_r_valid)
            gt_joints3d_l_valid_tf = rigid_tf_torch_batch(gt_joints3d_l_valid, R, t)

            # overwrite the left hand translation if there are two hands
            gt_cam_t_l[valid_idx] = (
                gt_joints3d_l_valid_tf[:, 0] - gt_root_cano_l[valid_idx]
            )

        if mode != "train":
            gt_vertices_l = gt_vertices_l + gt_cam_t_l[:, None, :]
        gt_model_joints_l = gt_model_joints_l + gt_cam_t_l[:, None, :]

        # camera translation for the two hands in euclidean space
        targets.register("mano.cam_t.r", gt_cam_t_r)
        targets.register("mano.cam_t.l", gt_cam_t_l)

        # convert perspective camera translation to a weak perspective camera
        avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
        gt_cam_t_wp_r = geometry.perspective_to_weak_perspective_torch(
            gt_cam_t_r, avg_focal_length, self.args.img_res
        )

        gt_cam_t_wp_l = geometry.perspective_to_weak_perspective_torch(
            gt_cam_t_l, avg_focal_length, self.args.img_res
        )

        # weak perspective camera translation
        targets.register("mano.cam_t.wp.r", gt_cam_t_wp_r)
        targets.register("mano.cam_t.wp.l", gt_cam_t_wp_l)

        # vertices and 3d joints in the image patch coordinate
        if mode != "train":
            targets.register("mano.vertices.cam.patch.r", gt_vertices_r)
            targets.register("mano.vertices.cam.patch.l", gt_vertices_l)
        targets.register("mano.joints3d.cam.patch.r", gt_model_joints_r)
        targets.register("mano.joints3d.cam.patch.l", gt_model_joints_l)
        return targets
