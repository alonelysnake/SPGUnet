nnUNetv2_plan_and_preprocess -d 119 -c 3d_fullres
nnUNetv2_train 119 3d_fullres 4
nnUNetv2_predict \
-i /root/autodl-tmp/nnUNet_raw/Dataset119_Vessel/imagesTs \
-o /root/nnUNet/nnUNet_results/Dataset119_Vessel/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/predict \
-d 119 -c 3d_fullres -f 4 -chk checkpoint_best.pth
nnUNetv2_predict \
-i /root/autodl-tmp/nnUNet_raw/Dataset119_Vessel/imagesTs \
-o /root/nnUNet/nnUNet_results/Dataset119_Vessel/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/predict \
-d 119 -c 3d_fullres -f 4 -chk checkpoint_best.pth
python mask_generate.py
