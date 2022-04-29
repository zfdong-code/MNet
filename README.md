# MNet

## MNet_pure
Implementations of MNet with MindSpore (https://www.mindspore.cn/) and PyTorch. 


## MNet_inserted_into_nnUNet
The proposed MNet is trained with nnUNet framework (https://github.com/MIC-DKFZ/nnUNet), thus we provide the whole modified nnUNet project.

Modifications we have done:
1) Add MNet.py and basic_module.py to /nnUNet/nnunet/network_architecture
2) Add myTrainer.py to /nnUNet/nnunet/training/network_training. 


Original training command: nnUNet_train 3d_fullres nnUNetTrainerV2 TaskXXX_MYTASK FOLD --npz
Current training command:  nnUNet_train 3d_fullres myTrainer TaskXXX_MYTASK FOLD --npz









<img src="https://github.com/zfdong-code/MNet/blob/main/MNet.jpg" width="800px"> 

# Dataset


