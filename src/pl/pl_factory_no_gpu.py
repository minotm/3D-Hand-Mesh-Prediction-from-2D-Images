from torch.utils.data import DataLoader
from src.models.ih_model import IHModel
from src.losses.ih_loss import IHLoss
from src.datasets.interhand_dataset import InterHandDataset
from src.datasets.test_dataset import TestDataset
from src.models.ih_wrapper_no_gpu import IHWrapper_no_gpu
from core.torch_utils import reset_all_seeds


def fetch_dataloader(args, mode, split, no_gt=False):
    dataset_id = "interhand5fps_220410"
    if no_gt:
        DATASET = TestDataset
    else:
        DATASET = InterHandDataset
    dataset = DATASET(dataset_id=dataset_id, split=split)
    collate_fn = None
    if mode == "train":
        reset_all_seeds(args.seed)
        return DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=args.shuffle_train,
            collate_fn=collate_fn,
        )

    elif mode in ["val", "test", "eval"]:
        return DataLoader(
            dataset=dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    else:
        assert False


def fetch_pytorch_model(args):
    model = IHModel(focal_length=args.focal_length, img_res=args.img_res)
    return model


def fetch_model(args):
    model = IHWrapper_no_gpu(args)
    return model


def fetch_loss_fn(hparams):
    loss_fn = IHLoss()
    return loss_fn
