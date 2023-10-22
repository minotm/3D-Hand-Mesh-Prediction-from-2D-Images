import random
import copy
import torch
import numpy as np
from loguru import logger

from src.datasets.dataset_utils import get_num_images
from src.utils.image_utils import read_img
from src.datasets.base_dataset import BaseDataset
from torchvision.transforms import Normalize
from src.utils.const import args
import core.sys_utils as sys_utils


def crop_kp2d(kp2d, cx, cy, dim):
    assert isinstance(kp2d, np.ndarray)
    assert len(kp2d.shape) == 2
    kp2d_cropped = np.copy(kp2d)
    kp2d_cropped[:, 0] -= cx - dim / 2
    kp2d_cropped[:, 1] -= cy - dim / 2
    return kp2d_cropped


def process_bbox(bbox):
    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.0
    c_y = bbox[1] + h / 2.0
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.0
    bbox[1] = c_y - bbox[3] / 2.0
    return bbox


def get_aug_intrix(fixed_focal: float, img_res):
    camera_center = np.array([img_res // 2, img_res // 2])
    intrx = torch.zeros([3, 3])
    intrx[0, 0] = fixed_focal
    intrx[1, 1] = fixed_focal
    intrx[2, 2] = 1.0
    intrx[0, -1] = camera_center[0]
    intrx[1, -1] = camera_center[1]
    return intrx


def load_ih_data(split, version):
    logger.info(f"Loading raw sessions: {split}")
    split_short = split.replace("mini", "").replace("tiny", "")
    split_p = f"/cluster/project/infk/hilliges/lectures/mp22/project3/mp_data/ih5fps_extras/interhand_{split_short}.npz"
    data = np.load(split_p)
    logger.info("Done")
    return data


def split_data_list(
    split,
    idx_list,
):

    num_samples = get_num_images(split, len(idx_list))
    random.seed(1)
    filtered_idx_list = copy.deepcopy(idx_list)
    random.shuffle(filtered_idx_list)
    final_idx_list = np.array(filtered_idx_list[:num_samples])
    return final_idx_list


class InterHandDataset(BaseDataset):
    def _load_data(self, split, use_augmentation):
        self.split = split
        self.is_train = split.endswith("train")
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

        if "train" in split:
            self.mode = "train"
        elif "val" in split:
            self.mode = "val"
        elif "test" in split:
            self.mode = "test"
        self.txn = sys_utils.fetch_lmdb_reader(
            f"/cluster/project/infk/hilliges/lectures/mp22/project3/mp_data/ih5fps_folders/interhand_{self.mode}.lmdb"
        )

        # load all data
        data = load_ih_data(split, self.version)
        idx_list = np.arange(len(data["imgname"]))
        idx_list = split_data_list(
            split=split,
            idx_list=idx_list,
        )

        imgname = data["imgname"]
        logger.info(
            f"{len(idx_list)} images are randomly sampled to obtain {self.split}"
        )

        # update new datalist based on idx_list
        self.imgname = imgname[idx_list]

        data_subset = {}
        for k in data.keys():
            data_subset[k] = data[k][idx_list]
        self.data = data_subset

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        self.cam_intrinsics = self.data["cam_intrinsics"].astype(np.float)

    def __init__(self, dataset_id, split, use_augmentation=True):
        self.version = dataset_id
        self._load_data(split, use_augmentation)

        # Get gt SMPL parameters, if available
        # hands
        self.pose_r = self.data["pose_r"].astype(np.float)
        self.betas_r = self.data["betas_r"].astype(np.float)
        self.joints3d_r = self.data["joints3d_r"].astype(np.float)
        self.joints2d_r = self.data["joints2d_r"].astype(np.float)

        self.pose_l = self.data["pose_l"].astype(np.float)
        self.betas_l = self.data["betas_l"].astype(np.float)
        self.joints3d_l = self.data["joints3d_l"].astype(np.float)
        self.joints2d_l = self.data["joints2d_l"].astype(np.float)
        self.bbox = self.data["bbox"].astype(np.float)

        # object
        logger.info(f"Loaded {self.split} split, num samples {len(self.imgname)}")

    def __getitem__(self, index):
        imgname = self.imgname[index].copy()

        # hands
        joints2d_r = self.joints2d_r[index].copy()
        joints3d_r = self.joints3d_r[index].copy()

        joints2d_l = self.joints2d_l[index].copy()
        joints3d_l = self.joints3d_l[index].copy()

        pose_r = self.pose_r[index].copy()
        betas_r = self.betas_r[index].copy()
        pose_l = self.pose_l[index].copy()
        betas_l = self.betas_l[index].copy()

        bbox = self.bbox[index].copy()
        bbox = process_bbox(bbox)
        bbox_w = bbox[2]
        bbox_h = bbox[3]
        cx = bbox[0] + bbox_w / 2.0
        cy = bbox[1] + bbox_h / 2.0
        bbox_dim = max(bbox_w, bbox_h)

        # image loading
        try:
            imgname = imgname.replace(
                f"./data/dataset_folders/{self.version}/images/", ""
            )
            cv_img = read_img(self.txn, imgname)
        except:
            logger.warning(f"Unable to load {imgname}")
            cv_img = np.zeros((1000, 400, 3), dtype=np.float32)

        # scale divided by 200 to make a smaller number for scale
        scale = bbox_dim / 200.0
        center = [cx, cy]

        # data augmentation: joints
        flip, pn, rot, sc = self.augm_params()

        # apply the same augmentation to the 2d joints
        # also normalize 2d joints
        joints2d_r = self.j2d_processing(joints2d_r, center, sc * scale, rot, flip)
        joints2d_l = self.j2d_processing(joints2d_l, center, sc * scale, rot, flip)

        # apply the same augmentation to the rgb images
        img = self.rgb_processing(
            cv_img, center, sc * scale, rot, flip, pn, img_res=args.img_res
        )
        img = torch.from_numpy(img).float()
        norm_img = self.normalize_img(img)

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}

        inputs["img"] = norm_img

        # hands
        # apply the augmentation to MANO poses
        targets["mano.pose.r"] = torch.from_numpy(
            self.pose_processing(pose_r, rot, flip)
        ).float()
        targets["mano.pose.l"] = torch.from_numpy(
            self.pose_processing(pose_l, rot, flip)
        ).float()
        targets["mano.beta.r"] = torch.from_numpy(betas_r).float()
        targets["mano.beta.l"] = torch.from_numpy(betas_l).float()
        targets["mano.joints2d.norm.r"] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets["mano.joints2d.norm.l"] = torch.from_numpy(joints2d_l[:, :2]).float()

        # full image camera coord
        targets["mano.joints3d.cam.full.r"] = torch.FloatTensor(joints3d_r[:, :3])
        targets["mano.joints3d.cam.full.l"] = torch.FloatTensor(joints3d_l[:, :3])

        intrx = get_aug_intrix(args.focal_length, args.img_res)
        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        meta_info["imgname"] = imgname
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["is_flipped"] = flip
        meta_info["rot_angle"] = np.float32(rot)  # amount of rotation
        meta_info["sample_index"] = index

        # whether the hand has GT in this image
        targets["left_valid"] = float(not torch.isnan(targets["mano.pose.l"].sum()))
        targets["right_valid"] = float(not torch.isnan(targets["mano.pose.r"].sum()))
        return inputs, targets, meta_info

    def __len__(self):
        return len(self.imgname)
