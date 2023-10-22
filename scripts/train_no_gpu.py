import comet_ml
import os
import sys
import torch
import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pprint import pformat
import os.path as op

sys.path.append(".")

from src.utils.const import args
import src.pl.pl_factory as pl_factory
from src.pl.pl_model_no_gpu import ModelPL_no_gpu
import core.comet_utils as comet_utils
from core.torch_utils import reset_all_seeds
import core.sys_utils as sys_utils


def init_experiment(args):
    api_key = os.environ["AOHMR_API_KEY"]
    workspace = os.environ["AOHMR_WORKSPACE"]
    project_name = "aohmr"

    if args.resume_ckpt == "":
        # Initialize comet cloud logger
        experiment = comet_ml.Experiment(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name,
            disabled=args.mute,
            display_summary_level=0,
        )
        exp_key = comet_utils.fetch_key_from_experiment(experiment)
    else:
        meta = torch.load(
            op.join("/".join(args.resume_ckpt.split("/")[:-2]), "meta.pt")
        )
        prev_key = meta["comet_key"]
        experiment = comet_ml.ExistingExperiment(
            previous_experiment=prev_key,
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
            disabled=args.mute,
            display_summary_level=0,
        )
        exp_key = prev_key[:9]
    args.exp_name = exp_key
    args.log_dir = f"./logs/{args.exp_name}"
    if args.resume_ckpt == "":
        os.makedirs(args.log_dir, exist_ok=True)
        meta_info = {"comet_key": experiment.get_key()}
        torch.save(meta_info, op.join(args.log_dir, "meta.pt"))
        tags = ['init']
        logger.info(f"Experiment tags: {tags}")
        experiment.add_tags(tags)
    return experiment, args


def main(args):

    reset_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #args.gpu = torch.cuda.get_device_properties(device).name
    args.gpu = None

    experiment, args = init_experiment(args)
    log_dir = args.log_dir

    logger.add(
        os.path.join(log_dir, "train.log"),
        level="INFO",
        colorize=False,
    )
    #logger.info(torch.cuda.get_device_properties(device))

    reset_all_seeds(args.seed)
    model = ModelPL_no_gpu(args).to(device)
    model.experiment = experiment

    ckpt_callback = ModelCheckpoint(
        monitor="loss",
        verbose=True,
        save_top_k=10,
        mode="min",
        every_n_epochs=args.eval_every_epoch,
    )

    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

    model_summary_cb = ModelSummary(max_depth=3)
    callbacks = [ckpt_callback, pbar_cb, model_summary_cb]
    trainer = pl.Trainer(
        gradient_clip_val=args.grad_clip,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.acc_grad,
        gpus=0,
        logger=None,
        min_epochs=args.num_epoch,
        max_epochs=args.num_epoch,
        callbacks=callbacks,
        log_every_n_steps=args.log_every,
        default_root_dir=log_dir,
        check_val_every_n_epoch=args.eval_every_epoch,
        num_sanity_val_steps=0,
        enable_model_summary=False,
    )

    reset_all_seeds(args.seed)
    train_loader = pl_factory.fetch_dataloader(args, "train", args.trainsplit)
    val_loader = pl_factory.fetch_dataloader(args, "val", args.valsplit)

    logger.info(f"Hyperparameters: \n {pformat(args)}")
    logger.info("*** Started training ***")
    reset_all_seeds(args.seed)
    experiment.log_parameters(args)
    trainer.fit(model, train_loader, [val_loader], ckpt_path=args.resume_ckpt)


if __name__ == "__main__":
    main(args)
