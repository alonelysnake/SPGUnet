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
./Dataset120_Supervoxel
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

We use this for the second training stage for our decoder. While training the first stage, just remove the last channel in key `channel_names`. More details and rules of the filename and environment variable configuration can be seen at [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

### 2. Supervoxel clustering

Run the following command to generate all the results of supervoxel clustering.

```
cd slic
python slic.py
```

The result folder will be generate as follow:

```
./slic/
	|-- label_vis1000/
	|-- label_vis1000_loop1/
	|-- label_1000/
	|-- label_1000_loop1/
	|-- result1000/
	|-- result1000_loop1/
	|-- result_mat/
```

Put the folder `label_vis1000_loop1` to the dataset and set the filename to fit the rules above.

## Training

### 1. Encoder Training

We follow the same dataset preprocessing and training framework as in [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Using the script to get the calcification mask.

```
bash ./mask_generate.sh
```

Then use the following command provided by nnU-Net's framework to generate the plan in encoder training and train the model.

```
nnUNetv2_plan_and_preprocess -d 119 -c 3d_fullres
nnUNetv2_train 119 3d_fullres 4
```

### 2. Decoder Training

The decoder can be trained after finishing training the encoder. Put the best model `checkpoint_best.pth` in the previous stage at the folder `./pretrain` and run the script to train.

```
bash ./decoder_train.sh
```

## Test

After getting the calcification mask result and supervoxel clustering result, we can test by the script.

```
bash ./test.sh
```

## Acknowledgement

This repository is built based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) repository.
