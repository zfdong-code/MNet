#MNet: Rethinking 2D/3D Networks for Anisotropic Medical Image Segmentation

MNet is novel data-independent CNN segmentation architecture, which realizes adaptive 2D/3D path selection, thus being robust to varying anisotropic degrees of medical datasets.


X

## MNet_pure
Implementations of MNet with MindSpore (https://www.mindspore.cn/) and PyTorch. 


## MNet_inserted_into_nnUNet
The proposed MNet is trained with nnUNet framework (https://github.com/MIC-DKFZ/nnUNet), thus we provide the whole modified nnUNet project. 

--Modifications we have done:
1) Add MNet.py and basic_module.py to **/nnUNet/nnunet/network_architecture**
2) Add myTrainer.py to **/nnUNet/nnunet/training/network_training**

(You can insert most end-to-end networks into nnUNet like the way we did)

--Training cmd:
nnUNet_train 3d_fullres myTrainer TaskXXX_MYTASK FOLD --npz (see https://github.com/MIC-DKFZ/nnUNet for details)










<img src="https://github.com/zfdong-code/MNet/blob/main/MNet.jpg" width="800px"> 

# Dataset

1. LiTS: https://competitions.codalab.org/competitions/17094#learn_the_details-evaluation
2. KiTS: https://kits19.grand-challenge.org/data/
3. BraTS: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data
4. PROMISE: https://promise12.grand-challenge.org/

