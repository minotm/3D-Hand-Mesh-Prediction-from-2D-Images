import torch
import pytorch_lightning as pl
import core.pl_utils as pl_utils
import core.torch_utils as torch_utils
from core.ld_utils import ld2dl
import torch.optim as optim
from core.comet_utils import log_dict
from core.pl_utils import push_checkpoint_metric, avg_losses_cpu
import time
import json
import numpy as np
from pprint import pprint


class AbstractPL(pl.LightningModule):
    def __init__(
        self,
        args,
        get_model_pl_fn,
        visualize_all_fn,
        push_images_fn,
        tracked_metric,
        metric_init_val,
    ):
        super().__init__()
        self.experiment = None
        self.args = args
        self.tracked_metric = tracked_metric
        self.metric_init_val = metric_init_val

        # self.experiment = self.args.experiment
        self.model = get_model_pl_fn(args)
        # self.model.load_pretrained(self.args.load_from)
        self.started_training = False
        self.loss_dict_vec = []
        self.has_applied_decay = False
        self.visualize_all = visualize_all_fn
        self.push_images = push_images_fn
        self.vis_train_batches = []
        self.vis_val_batches = []

    def set_training_flags(self):
        self.started_training = True

    def load_from_ckpt(self, ckpt_path):
        sd = torch.load(ckpt_path)["state_dict"]
        print(self.load_state_dict(sd))

    def training_step(self, batch, batch_idx):
        self.set_training_flags()
        if len(self.vis_train_batches) < 1:
            self.vis_train_batches.append(batch)
        inputs, targets, meta_info = batch

        loss = self.model(inputs, targets, meta_info, "train")
        loss = {k: loss[k].mean().view(-1) for k in loss}
        total_loss = sum(loss[k] for k in loss)

        loss_dict = {"total_loss": total_loss, "loss": total_loss}
        loss_dict.update(loss)

        for k, v in loss_dict.items():
            if k != "loss":
                loss_dict[k] = v.detach()

        log_every = self.args.log_every
        self.loss_dict_vec.append(loss_dict)
        self.loss_dict_vec = self.loss_dict_vec[len(self.loss_dict_vec) - log_every :]
        if batch_idx % log_every == 0 and batch_idx != 0:
            running_loss_dict = avg_losses_cpu(self.loss_dict_vec)
            log_dict(
                self.experiment,
                running_loss_dict,
                step=self.global_step,
                postfix="__train",
            )
        return loss_dict

    def training_epoch_end(self, outputs):
        outputs = avg_losses_cpu(outputs)
        self.experiment.log_epoch_end(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        if len(self.vis_val_batches) < 2:
            self.vis_val_batches.append(batch)
        out = self.eval_step(batch, batch_idx)
        return out

    def validation_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, postfix="__val")

    def test_step(self, batch, batch_idx):
        if self.mode == 'inference':
            return self.inf_step(batch, batch_idx)
        else:
            out = self.eval_step(batch, batch_idx)
            return out

    def inf_step(self, batch, batch_idx):
        if self.training:
            self.eval()
        with torch.no_grad():
            inputs, targets, meta_info = batch
            out = self.model(inputs, targets, meta_info, "inference")
            return {"out_dict": out}

    def test_epoch_end(self, outputs):
        if self.mode == 'inference':
            self.inference_epoch_end(outputs, '__test')
            return

        result, metrics, metric_dict = self.eval_epoch_end(
            outputs, postfix="__test"
        )
        for k, v in metric_dict.items():
            metric_dict[k] = float(v)

        # dump image names
        if self.interface_p is not None:
            imgnames = result["interface.meta_info.imgname"]
            with open(
                self.interface_p.replace(".params.pt", ".imgnames.json"), "w"
            ) as fp:
                json.dump({"imgname": imgnames}, fp, indent=4)

            torch.save(result, self.interface_p, pickle_protocol=4)

            print(f"Results: {self.interface_p}")

        if self.metric_p is not None:
            torch.save(metrics, self.metric_p)
            json_p = self.metric_p.replace(".pt", ".json")
            with open(json_p, "w") as f:
                json.dump(metric_dict, f, indent=4)
            print(f"Metrics: {self.metric_p}")
            print(f"Metric dict: {json_p}")

        return result

    def eval_step(self, batch, batch_idx):

        if self.training:
            self.eval()
        with torch.no_grad():
            inputs, targets, meta_info = batch
            out, loss = self.model(inputs, targets, meta_info, "test")
            return {"out_dict": out, "loss": loss}

    def inference_epoch_end(self, out_list, postfix):
        # do not compare with GT; only inference
        out_list_dict = ld2dl(out_list)
        outputs = ld2dl(out_list_dict["out_dict"])

        for k, tensor in outputs.items():
            if isinstance(tensor[0], list):
                outputs[k] = sum(tensor, [])
            else:
                outputs[k] = torch.cat(tensor)

        outputs = {k: torch_utils.tensor2np(v) for k, v in outputs.items()}

        out_dict = {}
        imgnames = outputs['imgname']
        joints3d_r = outputs['mano.joints3d.cam.patch.r']
        joints3d_l = outputs['mano.joints3d.cam.patch.l']
        for idx, key in enumerate(imgnames):
            j3d_r = joints3d_r[idx].tolist()
            j3d_l = joints3d_l[idx].tolist()
            assert key not in out_dict.keys()
            out_dict[key] = {}
            out_dict[key]['right'] = j3d_r
            out_dict[key]['left'] = j3d_l

        with open(self.save_p, 'w') as f:
            json.dump(out_dict, f, indent=4)
        print(f'Results saved at {self.save_p}')


    def eval_epoch_end(self, out_list, postfix):

        if not self.started_training:
            self.started_training = True
            result = push_checkpoint_metric(self.tracked_metric, self.metric_init_val)
            return result

        # unpack
        outputs, loss_dict = pl_utils.reform_outputs(out_list)

        if "test" in postfix:
            per_img_metric_dict = {}
            imgnames = outputs["interface.meta_info.imgname"]
            for k, v in outputs.items():
                if "metric." in k:
                    per_img_metric_dict[k] = np.array(v)

        metric_dict = {}
        for k, v in outputs.items():
            if "metric." in k:
                metric_dict[k] = np.nanmean(np.array(v))
                # per_img_metric_dict[k] = np.array(v)

        loss_metric_dict = {}
        loss_metric_dict.update(metric_dict)
        loss_metric_dict.update(loss_dict)
        pprint(metric_dict)

        if self.experiment is not None:
            log_dict(
                self.experiment,
                loss_metric_dict,
                step=self.global_step,
                postfix=postfix,
            )

            result = push_checkpoint_metric(
                self.tracked_metric, loss_metric_dict[self.tracked_metric]
            )
            self.log(self.tracked_metric, result[self.tracked_metric])

        if not self.args.no_vis:
            print("Rendering images")
            self.visualize_batches(self.vis_train_batches, "_train", 2, None)
            self.visualize_batches(self.vis_val_batches, "_val", 2, None)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, self.args.lr_dec_epoch, gamma=self.args.lr_decay, verbose=True
        )
        return [optimizer], [scheduler]

    def visualize_batches(self, batches, postfix, num_examples, no_tqdm=True):
        im_list = []
        if self.training:
            self.eval()

        tic = time.time()
        for batch_idx, batch in enumerate(batches):
            with torch.no_grad():
                inputs, targets, meta_info = batch
                vis_dict = self.model(inputs, targets, meta_info, "vis")
                curr_im_list = self.visualize_all(
                    vis_dict,
                    num_examples,
                    self.model.renderer,
                    postfix=postfix,
                    no_tqdm=no_tqdm,
                )
                self.push_images(self.experiment, curr_im_list, self.global_step)
                im_list += curr_im_list
                print("Rendering: %d/%d" % (batch_idx + 1, len(batches)))

        print("Done rendering (%.1fs)" % (time.time() - tic))
        return im_list
