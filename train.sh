nnUNet_preprocessed="/root/autodl-tmp/nnUNet_preprocessed"
nnUNet_results="/root/nnUNet/nnUNet_results"
nnUNet_raw="/root/autodl-tmp/nnUNet_raw"

export nnUNet_preprocessed="$nnUNet_preprocessed"
export nnUNet_results="$nnUNet_results"
export nnUNet_raw="$nnUNet_raw"

python train.py 120 4 -tr SPGUNetTrainer -p SPGUNetPlans -stage 0