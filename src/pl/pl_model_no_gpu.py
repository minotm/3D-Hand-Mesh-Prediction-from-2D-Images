from core.abstract_pl import AbstractPL
import src.pl.pl_factory_no_gpu as pl_factory_no_gpu
from core.comet_utils import push_images
from src.pl.visualize_pl_ih import visualize_all as visualize_all_ih


class ModelPL_no_gpu(AbstractPL):
    def __init__(self, args):
        vis_fn = visualize_all_ih
        super().__init__(
            args,
            pl_factory_no_gpu.fetch_model,
            vis_fn,
            push_images,
            "loss",
            float("inf"),
        )
