from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from dynamic_network_architectures.architectures.spgunet import SPGUNet
from nnunetv2.preprocessing.resampling.supervoxel_resampling import spgu_resample_data_or_seg_to_shape

from typing import List, Union, Tuple, Type

# plan for SPGUNet
class SPGUNetPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'SPGUNetPreprocessor',
                 plans_name: str = 'SPGUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id,
                         gpu_memory_target_in_gb=gpu_memory_target_in_gb,
                         preprocessor_name=preprocessor_name,
                         plans_name=plans_name,
                         overwrite_target_spacing=overwrite_target_spacing,
                         suppress_transpose=suppress_transpose)

        self.UNet_class = SPGUNet

    def determine_resampling(self, *args, **kwargs):
        """
                returns what functions to use for resampling data and seg, respectively. Also returns kwargs
                resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

                determine_resampling is called within get_plans_for_configuration to allow for different functions for each
                configuration
        """
        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs=super().determine_resampling(*args,**kwargs)
        resampling_data = spgu_resample_data_or_seg_to_shape
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs


if __name__ == '__main__':
    SPGUNetPlanner(2, 8).plan_experiment()