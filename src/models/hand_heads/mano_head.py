import torch.nn as nn
from src.utils.mano import build_mano
import src.utils.conversions as conversions
import core.ld_utils as ld_utils
import src.utils.geometry as geometry
from core.py3d import matrix_to_axis_angle
from src.utils.mano import build_mano
from core.unidict import unidict
import core.torch_cam_utils as tcu
from src.utils.mano import mano_layer


class MANOHead(nn.Module):
    def __init__(self, is_rhand, focal_length, img_res):
        super(MANOHead, self).__init__()
        hand_type = "right" if is_rhand else "left"
        self.mano = mano_layer[hand_type]
        self.add_module("mano", self.mano)
        self.focal_length = focal_length
        self.img_res = img_res
        self.is_rhand = is_rhand

    def forward(self, rotmat, shape, cam, K):
        rotmat_original = rotmat.clone()
        rotmat = matrix_to_axis_angle(rotmat.reshape(-1, 3, 3)).reshape(-1, 48)

        mano_output = self.mano(
            betas=shape,
            hand_pose=rotmat[:, 3:],
            global_orient=rotmat[:, :3],
        )
        output = unidict()
        avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
        cam_t = geometry.weak_perspective_to_perspective_torch(
            cam,
            focal_length=avg_focal_length,
            img_res=self.img_res,
        )

        joints3d_cam = mano_output.joints + cam_t[:, None, :]
        v3d_cam = mano_output.vertices + cam_t[:, None, :]

        joints2d = tcu.project2d_batch(K, joints3d_cam)
        joints2d = conversions.normalize_kp2d(joints2d, self.img_res)

        output.register("cam_t.wp", cam)
        output.register("cam_t", cam_t)
        output.register("joints3d", mano_output.joints)
        output.register("vertices", mano_output.vertices)
        output.register("joints3d.cam.patch", joints3d_cam)
        output.register("vertices.cam.patch", v3d_cam)
        output.register("joints2d.norm", joints2d)
        output.register("beta", shape)
        output.register("pose", rotmat_original)

        postfix = ".r" if self.is_rhand else ".l"
        output_pad = output.postfix(postfix)
        return output_pad
