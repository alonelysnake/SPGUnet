import math
from typing import Union

import torch
from torch import autocast
import torch.nn.functional as F
from torch._dynamo import OptimizedModule
import os
from batchgenerators.utilities.file_and_folder_operations import join, isfile
import sys
import numpy as np

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.helpers import empty_cache
from dynamic_network_architectures.architectures.spgunet import SPGUNet


class SPGUNetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.stage = 0
        self.param_only = False

    def initialize(self):
        self.print_to_log_file('use spgu trainer')
        super().initialize()

    def load_checkpoint(self, filename_or_checkpoint: str, param_only: bool = False) -> None:
        if not param_only:
            super().load_checkpoint(filename_or_checkpoint)
        else:
            if not self.was_initialized:
                self.initialize()

            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)

            new_state_dict = {}
            for k, value in checkpoint['network_weights'].items():
                key = k
                if key not in self.network.state_dict().keys() and key.startswith('module.'):
                    key = key[7:]
                new_state_dict[key] = value

            # messing with state dict naming schemes. Facepalm.
            if self.is_ddp:
                if isinstance(self.network.module, OptimizedModule):
                    self.network.module._orig_mod.load_state_dict(new_state_dict)
                else:
                    self.network.module.load_state_dict(new_state_dict)
            else:
                if isinstance(self.network, OptimizedModule):
                    self.network._orig_mod.load_state_dict(new_state_dict)
                else:
                    self.network.load_state_dict(new_state_dict)

    def set_stage(self, stage):
        self.stage = stage

    def on_train_end(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        if self.stage==1:
            super(SPGUNetTrainer, self).on_train_end()
        else:
            self.current_epoch -= 1
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
            self.current_epoch += 1

            # shut down dataloaders
            old_stdout = sys.stdout
            with open(os.devnull, 'w') as f:
                sys.stdout = f
                if self.dataloader_train is not None:
                    self.dataloader_train._finish()
                if self.dataloader_val is not None:
                    self.dataloader_val._finish()
                sys.stdout = old_stdout

            empty_cache(self.device)
            self.print_to_log_file("Training first stage done.")
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.decoder.deep_supervision = enabled
            self.network.module.final_decoder.deep_supervision = enabled
        else:
            self.network.decoder.deep_supervision = enabled
            self.network.final_decoder.deep_supervision = enabled

    def run_training(self):
        if not self.was_initialized:
            self.initialize()
        assert isinstance(self.network, SPGUNet)
        self.network.set_stage(self.stage)
        super(SPGUNetTrainer, self).run_training()


if __name__ == "__main__":
    tr = SPGUNetTrainer(plans={}, configuration="3d_fullres", fold=0,
                        dataset_json={}, unpack_dataset=False)
