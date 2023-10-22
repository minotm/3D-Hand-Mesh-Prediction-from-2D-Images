from smplx import MANO
from src.utils.const import args
import os
import smplx
import torch


def build_mano(is_rhand):
    return MANO(
        args.mano_model_dir,
        create_transl=False,
        use_pca=False,
        flat_hand_mean=False,
        is_rhand=is_rhand,
    )


mano_path_r = os.environ["MANO_MODEL_DIR_R"]
mano_path_l = os.environ["MANO_MODEL_DIR_L"]
mano_layer = {
    "right": smplx.create(mano_path_r, "mano", use_pca=False, is_rhand=True),
    "left": smplx.create(mano_path_l, "mano", use_pca=False, is_rhand=False),
}

# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if (
    torch.sum(
        torch.abs(
            mano_layer["left"].shapedirs[:, 0, :]
            - mano_layer["right"].shapedirs[:, 0, :]
        )
    )
    < 1
):
    print("Fix shapedirs bug of MANO")
    mano_layer["left"].shapedirs[:, 0, :] *= -1
