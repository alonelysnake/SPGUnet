nnUNet_preprocessed="/root/autodl-tmp/nnUNet_preprocessed"
nnUNet_results="/root/nnUNet/nnUNet_results"
nnUNet_raw="/root/autodl-tmp/nnUNet_raw"

data_dir="$nnUNet_raw/Dataset119_Vessel"
pred_dir="$nnUNet_results/Dataset119_Vessel/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/predict"

export nnUNet_preprocessed="$nnUNet_preprocessed"
export nnUNet_results="$nnUNet_results"
export nnUNet_raw="$nnUNet_raw"

nnUNetv2_plan_and_preprocess -d 119 -c 3d_fullres
nnUNetv2_train 119 3d_fullres 4
nnUNetv2_predict \
-i "$data_dir/imagesTr" \
-o "$pred_dir/imagesTr" \
-d 119 -c 3d_fullres -f 4 -chk checkpoint_best.pth
nnUNetv2_predict \
-i "$data_dir/imagesTs" \
-o "$pred_dir/imagesTs" \
-d 119 -c 3d_fullres -f 4 -chk checkpoint_best.pth

python preprocess.py -d 120 -mask_p $pred_dir -overwrite_plans_name SPGUNetPlans -pl SPGUNetPlanner