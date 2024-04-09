nnUNetv2_plan_and_preprocess -d 120 -c 3d_fullres -pl SupervoxelPlanner
nnUNetv2_train 119 3d_fullres 4 -p SupervoxelPlans --c