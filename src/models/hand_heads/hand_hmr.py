import torch
import torch.nn as nn
from src.models.hmr_layer import HMRLayer
from core.py3d import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix
from core.unidict import unidict


class HandHMR(nn.Module):
    def __init__(self, feat_dim, is_rhand, n_iter):
        super().__init__()
        self.is_rhand = is_rhand

        hand_specs = {"pose_6d": 6 * 16, "cam_t/wp": 3, "shape": 10}
        self.hmr_layer = HMRLayer(feat_dim, 1024, hand_specs)

        self.cam_init = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.hand_specs = hand_specs
        self.n_iter = n_iter
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def init_vector_dict(self, features):
        batch_size = features.shape[0]
        dev = features.device
        init_pose = (
            matrix_to_rotation_6d(
                axis_angle_to_matrix(torch.zeros(16, 3))
            )
            .reshape(1, -1)
            .repeat(batch_size, 1)
        )
        init_shape = torch.zeros(1, 10).repeat(batch_size, 1)
        init_transl = self.cam_init(features)

        out = {}
        out["pose_6d"] = init_pose
        out["shape"] = init_shape
        out["cam_t/wp"] = init_transl
        out = unidict(out).to(dev)
        return out

    def forward(self, feat):
        batch_size = feat.shape[0]
        feat = feat.view(feat.size(0), -1)

        init_vdict = self.init_vector_dict(feat)
        init_cam_t = init_vdict["cam_t/wp"].clone()
        pred_vdict = self.hmr_layer(feat, init_vdict, self.n_iter)

        pred_rotmat = rotation_6d_to_matrix(
            pred_vdict["pose_6d"].reshape(-1, 6)
        ).view(batch_size, 16, 3, 3)

        pred_vdict.register("pose", pred_rotmat)
        pred_vdict.register("cam_t.wp.init", init_cam_t)
        pred_vdict = pred_vdict.replace_keys("/", ".")
        return pred_vdict
