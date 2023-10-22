import cv2
import numpy as np
from torch.utils.data import Dataset
import src.utils.conversions as conversions
import src.datasets.dataset_utils as dataset_utils
import src.utils.image_utils as image_utils
from src.utils.const import args as options


class BaseDataset(Dataset):
    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0  # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0  # rotation
        sc = 1  # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= options.flip_prob:
                flip = 1
                assert False, "Flipping not supported"

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(
                1 - options.noise_factor, 1 + options.noise_factor, 3
            )

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(
                2 * options.rot_factor,
                max(
                    -2 * options.rot_factor,
                    np.random.randn() * options.rot_factor,
                ),
            )

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(
                1 + options.scale_factor,
                max(
                    1 - options.scale_factor,
                    np.random.randn() * options.scale_factor + 1,
                ),
            )
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, img_res):
        crop_dim = int(scale * 200)
        # faster cropping!!
        rgb_img = dataset_utils.generate_patch_image(
            rgb_img,
            [center[0], center[1], crop_dim, crop_dim],
            1.0,
            rot,
            [img_res, img_res],
            cv2.INTER_CUBIC,
        )[0]

        # flip the image
        if flip:
            rgb_img = image_utils.flip_img(rgb_img)

        assert flip != 1

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
        rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = image_utils.transform(
                kp[i, 0:2] + 1,
                center,
                scale,
                [options.img_res, options.img_res],
                rot=r,
            )
        # convert to normalized coordinates
        kp = conversions.normalize_kp2d_np(kp, options.img_res)
        # flip the x coordinates
        if f:
            kp = image_utils.flip_kp(kp)
        kp = kp.astype("float32")
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
        S[:, :-1] = np.einsum("ij,kj->ki", rot_mat, S[:, :-1])
        # flip the x coordinates
        if f:
            S = image_utils.flip_kp(S)
        S = S.astype("float32")
        return S

    def pose_processing(self, pose, r, f):
        """Process MANO theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = image_utils.rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = image_utils.flip_pose(pose)
        # (72),float
        pose = pose.astype("float32")
        return pose
