# PRML Project 3, Spring 2023

  /\ ___ /\
 (  o   o  ) Wecome to PRML Project 3!
  \  >#<  /
  /       \  
 /         \       ^
|           |     //
 \         /    //
  ///  ///   --

We provide some starter code for this project. Some functions are fully implemented, and there are some others that you will need to implement yourself. See below for file descriptions. Read the project description on CANVAS for more details on the project. Note the provided code is just a starting point and you may want to add certain functionality to it other than what is required.

## Installation
This project will use multiple packages which you will likely need to install. We recommend creating a virtual environment using Anaconda for this project and installing packages via pip. The required packages are listed in `requirements.txt`. The following demonstrates setting up a conda env and installing the required packages:

```
conda create -n p python=3.9
conda activate p
pip install -r requirements.txt
```

Please be aware the installation process can vary between operating systems. If you have a GPU, you may also want to install the GPU version of PyTorch. See the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for more details.

## Running the Code
The main file, `classification.py`, contains the code for running the classification. The code is set up to run with the default values for all arguments. You can vary the arguments by passing them in from the command line. For the command line arguments, please look at the `arg_parse` function in `util.py`. **Note**: to train the model you'll need to have the arg --train.

You will control which dataset to work with through `--dataset`: which dataset to use (taiji or wallpaper). Please feel free to modify the arguments in anyway you see fit.

## Quickstart 
After installing the required packages, you can run the code to train each "variant" of the datasets with (assuming your system is compatible with the default arguments):
```
python classification.py --dataset Taiji --train --fp_size full
python classification.py --dataset Taiji --train --fp_size lod4
python classification.py --dataset Wallpaper --train --test_set test
python classification.py --dataset Wallpaper --train --test_set test_challenge
```


## Data
- data/taiji_data_full.npz (same as P2)
  - The data you will be working with
  - Variables (np arrays):
    - **feature_names**: the names of all 1961 features
    - **form_names**: the names of all 45 forms you will be trying to classify
    - **labels**: provided class labels corresponding to each data point
    - **sub_info**: information about subject number and performance number corresponding to each data point
    - **data**: the actual data you will be working with
- data/taiji_data_lod4_fp.npz (same as taiji_data_full except with downsampled foot pressure)

- data/wallpapes
    - tain and test directories containing the images for the wallpaper dataset
    - Both train and test have directories for each class containing the images for that class.
    - test_challenge contains the augmented test images (harder)

## Provided Code
Note this data is just to get you started and you may edit it according to your needs so long as the basic functionality is preserved. You'll be given code to
- train and test the model
- load the datasets and create a dataloader
- Some of the visualization code such as plotting the confusion matrix and accuracy

## Code you need to edit/implement
Note that most (if not all) places you need to edit the code are marked with "TODO" markers that you can search for.
- A function for visualizing a networks layers
- A function for performing t-SNE on the features
- CNN + Multi-Layer Perceptron (MLP) imrpovements
- A function for performing data augmentation (EC)
- A function for finetuning a pretrained model (EC)

## Package Imports for the Assigment
Everything you need is in the starter code for the MLP.
For the CNN impelementation, you will make use of the following packages:
- Data augmentation: `torchvision.transforms`
- Finetuning an existing pretrained model: `torchvision.models`
- t-SNE: `sklearn.manifold.TSNE`
- Layer visualization: can be done using the existing packages (torchvision, torch, etc.)


## Tips
- Read the project description on CANVAS for more details on the project.
- You may find it helpful to save the trained model and load it for visualization or testing purposes. You will need to add functionality to support this.
- The starter code is just a starting point and you may want to add certain functionality. You can add helper functions, modify exsiting functions as long as the underlying functionality is preserved, etc. (as long as no external packages are used other than what we are already using)

Good luck!
