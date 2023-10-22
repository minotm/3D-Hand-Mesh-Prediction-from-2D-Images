import torch
import numpy as np
from loguru import logger
import core.sys_utils as sys_utils
from torchvision.transforms import Normalize
from src.datasets.base_dataset import BaseDataset
from src.utils.const import args
from src.datasets.interhand_dataset import load_ih_data, split_data_list
from src.datasets.interhand_dataset import get_aug_intrix, process_bbox
from src.utils.image_utils import read_img


class TestDataset(BaseDataset):
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
            f"/cluster/project/infk/hilliges/lectures/mp22/project3/mp_data/ih5fps_folders/interhand_{self.mode}.lmdb")

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
        self.bbox = self.data["bbox"].astype(np.float)
        logger.info(f"Loaded {self.split} split, num samples {len(self.imgname)}")

    def __getitem__(self, index):
        imgname = self.imgname[index].copy()
        bbox = self.bbox[index].copy()
        bbox = process_bbox(bbox)

        bbox_w = bbox[2]
        bbox_h = bbox[3]
        cx = bbox[0] + bbox_w / 2.0
        cy = bbox[1] + bbox_h / 2.0
        bbox_dim = max(bbox_w, bbox_h)

        # image loading
        try:
            cv_img = read_img(self.txn, imgname)
        except:
            logger.warning(f"Unable to load {imgname}")
            cv_img = np.zeros((1000, 400, 3), dtype=np.float32)

        scale = bbox_dim / 200.0
        center = [cx, cy]

        # disabled augmentation
        flip, pn, rot, sc = self.augm_params()

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
        meta_info["imgname"] = imgname
        intrx = get_aug_intrix(args.focal_length, args.img_res)
        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["sample_index"] = index
        return inputs, targets, meta_info

    def __len__(self):
        return len(self.imgname)
