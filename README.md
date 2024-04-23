# SPGUnet
Code for Spatial Prior-Guided UNet (SPGUnet)

## Requirements

Run the following command to install the required packages:

```
cd dynamic-network-architectures
pip install -e .
cd ../nnUNet
pip install -e .
```

## Preprocess

### 1. Dataset structure

We follow the same structure as in [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). The dataset structure should be as follows:

```
./Dataset119_Vessel
	|-- imagesTr/
	|-- imagesTs/
	|-- labelsTr/
	|-- labelsTs/
	|-- dataset.json
./Dataset120_Vessel
	|-- imagesTr/
	|-- imagesTs/
	|-- labelsTr/
	|-- labelsTs/
	|-- dataset.json
```

As for the information of dataset in `dataset.json`, the contents are as follows:

```
{
    "channel_names": {
        "0": "CT",
        "1": "cal",
        "2": "supervoxel"
    },
    "labels": {
        "background": 0,
        "vessel": 1,
        "cal": 2,
        "lumen": 3
    },
    "numTraining": 130,
    "file_ending": ".nii.gz"
}

```

We use this for the directory of `Dataset120_Vessel`. Just keep the first channel in key `channel_names` in the directory of `Dataset119_Vessel`. More details and rules of the filename and environment variable configuration can be seen at [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

Our dataset is currently not publicly available, but may be in the future. Please stay tuned.

### 2. Preprocessing

We follow the same dataset preprocessing and training framework as in [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).  The `preprocess.sh` contains all commands that we need. Run it to get the preprocessed data that used in training stage.

```
bash preprocess.sh
```

If you only want to get the supervoxel result, you can run the following command to generate the results of supervoxel clustering that we use.

```
cd slic
python slic.py
```

## Training

### 1. Encoder Training

Run the `train.sh` to train our network. If you have set the environment of nnU-Net before, we recommand you to use the following command to run manually. The command will run both stages automatically. If you interrupt the training but want to continue, add the parameter `--c`.

```
python train.py 120 4 -tr SPGUNetTrainer -p SPGUNetPlans -stage 0
```

### 2. Decoder Training

If you have trained the encoder and only want to train the decoder, use following command to skip the first stage.

```
python train.py 120 4 -tr SPGUNetTrainer -p SPGUNetPlans -stage 1
```

## Test

After getting the calcification mask result and supervoxel clustering result, we can test by the script.

```
bash ./test.sh
```

## Acknowledgement

This repository is built based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) repository.
