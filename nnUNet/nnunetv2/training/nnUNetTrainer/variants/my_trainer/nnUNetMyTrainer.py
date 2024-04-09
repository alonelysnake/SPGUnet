import math

import torch
from torch import autocast
import torch.nn.functional as F
import numpy as np

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.helpers import dummy_context


class nnUNetMyTrainer(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.sample_num = 20000
        self.lesion_weight = 0.5

    def initialize(self):
        self.print_to_log_file('use my trainer')
        super(nnUNetMyTrainer, self).initialize()

    def sample(self, shape, target: torch.Tensor = None):
        if target is not None:
            # batch_size = target.shape[0]
            coordinates = []
            for t in target:
                t = t.squeeze(0)
                coordinates_lesion = torch.from_numpy(np.argwhere(t.numpy() > 0))
                vol_shape = target.shape[2:]
                coordinates_lesion = coordinates_lesion.float() / torch.Tensor(list(vol_shape))
                if coordinates_lesion.shape[0] > int(self.sample_num * self.lesion_weight):
                    coordinates_lesion = coordinates_lesion[torch.randperm(coordinates_lesion.shape[0])[:int(self.sample_num * self.lesion_weight)]]

                samples_lesion = coordinates_lesion.shape[0]
                samples_no_lesion = self.sample_num - samples_lesion

                coordinates_lesion = coordinates_lesion[torch.randperm(coordinates_lesion.shape[0])]
                coordinates_no_lesion = (torch.rand(samples_no_lesion, 3))
                # Concat coordinates
                coordinates_batch = torch.cat([coordinates_lesion, coordinates_no_lesion], dim=0)  # shape=[sample_num,3]
                # Shuffle coordinates
                coordinates_batch = coordinates_batch[torch.randperm(coordinates_batch.shape[0])]
                coordinates.append(coordinates_batch)
            coordinates = torch.stack(coordinates).unsqueeze(1).unsqueeze(1)
            return coordinates

        batch_size = shape[0]

        random_depth = (torch.rand(self.sample_num))
        random_height = (torch.rand(self.sample_num))
        random_width = (torch.rand(self.sample_num))

        random_coordinates = torch.stack([random_depth, random_height, random_width], dim=1)
        random_coordinates = torch.stack([random_coordinates for i in range(batch_size)], dim=0)
        random_coordinates = random_coordinates.unsqueeze(1).unsqueeze(1)  # shape = [b,1,sample_num,3]

        return random_coordinates  # shape = [b,1,1,sample_num,3]

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        sample_points = self.sample(data.shape,target=target)
        target = F.grid_sample(target, sample_points, mode='nearest')
        sample_points = sample_points.to(self.device, non_blocking=True)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)


        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, points=sample_points)
            # del data
            output = output.unsqueeze(2).unsqueeze(2)  # output.shape=[b,num_class,sample_num]->[b,class,1,1,sample_num]

            # target.shape=[b,1,1,1,sample_num]
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        sample_points = self.sample(data.shape,target=target)
        target = F.grid_sample(target, sample_points, mode='nearest')
        sample_points = sample_points.to(self.device, non_blocking=True)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, points=sample_points)
            del data
            output = output.unsqueeze(2).unsqueeze(2)  # output.shape=[b,num_class,sample_num]->[b,class,1,1,sample_num]
            l = self.loss(output, target)

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
