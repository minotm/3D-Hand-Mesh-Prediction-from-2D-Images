import torch
import pytorch_lightning as pl
import sys

sys.path.append(".")

from src.pl.pl_model import ModelPL
from src.utils.const import args
import src.pl.pl_factory as pl_factory


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        gpus=1, logger=None, num_sanity_val_steps=0, enable_model_summary=False
    )
    model = ModelPL(args).to(device)
    model.started_training = True
    model.eval()
    model.mode = "inference"
    model.save_p = args.load_ckpt.split("/")[1] + ".json"

    # force eval split
    args.valsplit = "test"
    val_loader = pl_factory.fetch_dataloader(args, "test", "test", no_gt=True)
    trainer.test(model, ckpt_path=args.load_ckpt, dataloaders=[val_loader])


if __name__ == "__main__":
    main(args)
