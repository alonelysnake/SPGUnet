from default_experiment_planner import ExperimentPlanner
from dynamic_network_architectures.architectures.mynet import MySuperVoxelDecoderUNet

from typing import List, Union, Tuple, Type

# ljh 用于生成supervoxeldecoder的plan
class SupervoxelPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'SupervoxelPreprocessor',
                 plans_name: str = 'SupervoxelPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id,
                         gpu_memory_target_in_gb=gpu_memory_target_in_gb,
                         preprocessor_name=preprocessor_name,
                         plans_name=plans_name,
                         overwrite_target_spacing=overwrite_target_spacing,
                         suppress_transpose=suppress_transpose)

        self.UNet_class = MySuperVoxelDecoderUNet


if __name__ == '__main__':
    SupervoxelPlanner(2, 8).plan_experiment()