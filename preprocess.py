from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.paths import nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join
from slic.slic import slic_process

from glob import glob
import os
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

import argparse


def mask_generate(pred_dir, mask_dir):
    for c in ['imagesTr', 'imagesTs']:
        for nm in os.listdir(os.path.join(pred_dir, c)):
            if '.nii.gz' not in nm:
                continue
            pred_path = os.path.join(pred_dir, c, nm)
            mask_path = os.path.join(mask_dir, c, nm.replace('.nii.gz', '_0001.nii.gz'))

            vol = sitk.ReadImage(pred_path)
            v = sitk.GetArrayFromImage(vol)
            v = np.where(v == 2, 1, 0)
            v = sitk.GetImageFromArray(v)
            v.SetOrigin(vol.GetOrigin())
            v.SetSpacing(vol.GetSpacing())
            v.SetDirection(vol.GetDirection())
            sitk.WriteImage(v, mask_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-mask_p', type=str,
                        help="[REQUIRED] predict path of nnunet")
    parser.add_argument('-test_only', required=False, action="store_true",
                        help="[OPTIONAL] only preprocess for test")
    parser.add_argument('-fpe', type=str, required=False, default='DatasetFingerprintExtractor',
                        help='[OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is '
                             '\'DatasetFingerprintExtractor\'.')
    parser.add_argument('-npfp', type=int, default=8, required=False,
                        help='[OPTIONAL] Number of processes used for fingerprint extraction. Default: 8')
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="[RECOMMENDED] set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("--clean", required=False, default=False, action="store_true",
                        help='[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a '
                             'fingerprint already exists, the fingerprint extractor will not run. REQUIRED IF YOU '
                             'CHANGE THE DATASET FINGERPRINT EXTRACTOR OR MAKE CHANGES TO THE DATASET!')
    parser.add_argument('-pl', type=str, default='SPGUNetPlanner', required=False,
                        help='[OPTIONAL] Name of the Experiment Planner class that should be used. Default is '
                             '\'ExperimentPlanner\'. Note: There is no longer a distinction between 2d and 3d planner. '
                             'It\'s an all in one solution now. Wuch. Such amazing.')
    parser.add_argument('-gpu_memory_target', default=8, type=int, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom GPU memory target. Default: 8 [GB]. Changing this will '
                             'affect patch and batch size and will '
                             'definitely affect your models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')
    parser.add_argument('-preprocessor_name', default='DefaultPreprocessor', type=str, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom preprocessor class. This class must be located in '
                             'nnunetv2.preprocessing. Default: \'DefaultPreprocessor\'. Changing this may affect your '
                             'models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')
    parser.add_argument('-overwrite_target_spacing', default=None, nargs='+', required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom target spacing for the 3d_fullres and 3d_cascade_fullres '
                             'configurations. Default: None [no changes]. Changing this will affect image size and '
                             'potentially patch and batch '
                             'size. This will definitely affect your models performance! Only use this if you really '
                             'know what you are doing and NEVER use this without running the default nnU-Net first '
                             '(as a baseline). Changing the target spacing for the other configurations is currently '
                             'not implemented. New target spacing must be a list of three numbers!')
    parser.add_argument('-overwrite_plans_name', default='SPGUNetPlans', required=False,
                        help='[OPTIONAL] uSE A CUSTOM PLANS IDENTIFIER. If you used -gpu_memory_target, '
                             '-preprocessor_name or '
                             '-overwrite_target_spacing it is best practice to use -overwrite_plans_name to generate a '
                             'differently named plans file such that the nnunet default plans are not '
                             'overwritten. You will then need to specify your custom plans file with -p whenever '
                             'running other nnunet commands (training, inference etc)')
    parser.add_argument('-np', type=int, nargs='+', default=4, required=False)
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! '
                             'Recommended for cluster environments')
    args = parser.parse_args()
    return args


def plan_and_preprocess(args):
    data_dir = join(nnUNet_raw, convert_id_to_dataset_name(args.d[0]))

    # supervoxel clustering
    print("cluster supervoxel")
    paths = glob(join(data_dir, 'imagesTr', '*_0000.nii.gz'))
    paths.extend(glob(join(data_dir, 'imagesTs', '*_0000.nii.gz')))

    import multiprocessing as mp
    pool = mp.Pool(processes=16)
    for _ in tqdm(pool.imap_unordered(slic_process, paths),total=len(paths)):
        pass
    pool.close()
    pool.join()
    print("cluster finish")

    # mask generate
    mask_root_dir = args.mask_p
    mask_generate(mask_root_dir, data_dir)

    if not args.test_only:
        # fingerprint extraction
        print("Fingerprint extraction...")
        extract_fingerprints(args.d, args.fpe, args.npfp, args.verify_dataset_integrity, args.clean, args.verbose)

        # experiment planning
        print('Experiment planning...')
        plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing, args.overwrite_plans_name)

        # preprocessing
        print('Preprocessing...')
        # preprocess(args.d, args.overwrite_plans_name, args.c, args.np, args.verbose)
        preprocess(args.d, args.overwrite_plans_name, ('3d_fullres',), [args.np], args.verbose)


if __name__ == '__main__':
    args = parse_args()
    plan_and_preprocess(args)
