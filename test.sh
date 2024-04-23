nnUNet_preprocessed="/root/autodl-tmp/nnUNet_preprocessed"
nnUNet_results="/root/nnUNet/nnUNet_results"
nnUNet_raw="/root/autodl-tmp/nnUNet_raw"

data_dir="$nnUNet_raw/Dataset119_Vessel"
pred_dir="$nnUNet_results/Dataset120_Vessel/SPGUNetTrainer__SPGUNetPlans__3d_fullres/fold_4/predict"

export nnUNet_preprocessed="$nnUNet_preprocessed"
export nnUNet_results="$nnUNet_results"
export nnUNet_raw="$nnUNet_raw"

python test.py \
-i $nnUNet_raw/Dataset120_Vessel/imagesTs \
-o $pred_dir \
-d 120 -p SPGUNetPlans -tr SPGUNetTrainer -f 4 -chk checkpoint_best.pth