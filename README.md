## MNet_pure
Implementations of MNet with MindSpore (https://www.mindspore.cn/) and PyTorch. 


## MNet_inserted_into_nnUNet
The proposed MNet is trained with nnUNet framework (https://github.com/MIC-DKFZ/nnUNet), thus we provide the whole modified nnUNet project. 

Modifications we have done:
1) Add MNet.py and basic_module.py to **/nnUNet/nnunet/network_architecture**
2) Add myTrainer.py to **/nnUNet/nnunet/training/network_training**

(You can insert most end-to-end networks into nnUNet like the way we did)

Training cmd:
nnUNet_train 3d_fullres myTrainer TaskXXX_MYTASK FOLD --npz (see https://github.com/MIC-DKFZ/nnUNet for details)










<img src="https://github.com/zfdong-code/MNet/blob/main/MNet.jpg" width="800px"> 

# Dataset


