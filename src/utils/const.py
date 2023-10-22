from loguru import logger
import argparse


def construct_args_object():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainsplit",
        type=str,
        default="train",
        choices=["train", "minitrain"],
        help="Amount to subsample training set.",
    )
    parser.add_argument(
        "--valsplit",
        type=str,
        default="val",
        choices=["val", "minival"],
        help="Amount to subsample validation set.",
    )
    parser.add_argument("--log_every", type=int, default=5, help="log every k steps")
    parser.add_argument(
        "--eval_every_epoch", type=int, default=5, help="Eval every k epochs"
    )
    parser.add_argument(
        "--lr_dec_epoch",
        type=int,
        nargs="+",
        default=[],
        help="Learning rate decay epoch.",
    )
    parser.add_argument(
        "--load_ckpt", type=str, default="", help="Load checkpoints from PL format"
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--lr_decay", type=float, default=0.1, help="Learning rate decay factor"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "-f",
        "--fast",
        dest="fast_dev_run",
        help="single batch for development",
        action="store_true",
    )
    parser.add_argument("--no_vis", help="No vis", action="store_true")
    parser.add_argument("--mute", help="No logging", action="store_true")
    config = parser.parse_args()

    logger.info(f"Input arguments: \n {config}")

    if config.fast_dev_run:
        config.num_workers = 0
        config.batch_size = 8
        config.trainsplit = "minitrain"
        config.valsplit = "minival"
    return config


# static const
args = construct_args_object()
args.focal_length = 500.0
args.img_res = 224
args.rot_factor = 5.0
args.noise_factor = 0.1
args.scale_factor = 0.05
args.flip_prob = 0.0
args.img_norm_mean = [0.485, 0.456, 0.406]
args.img_norm_std = [0.229, 0.224, 0.225]
args.use_crop = True
args.pin_memory = True
args.shuffle_train = True
args.seed = 1
args.grad_clip = 100.0
args.acc_grad = 1
args.mano_model_dir = "data/body_models/mano"
