# Micro_Object-_Sim2Real
The official implementation of paper: Physics-Informed Machine Learning for Efficient Sim-to-Real Data Augmentation in Micro-Object Pose Estimation

Developers: Zongcai Tan and Lan Wei
Authors: Zongcai Tan, Lan Wei and Dandan Zhang

## Usage

First, install PyTorch == 1.9.1+cu111\\
prchvision == 0.10.1+cu111 \\
tensorboardx == 2.4\\
tensorboard_logger == 0.1.0:
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboardx == 2.4
pip install tensorboard_logger == 0.1.0
```

# Train Pixel GAN:
```
python train.py --checkpoints_dir ./xxx --batch_size 128 --dataroot "./xxx" --name PixGAN --model pix2pix --netG unet_256 --gan_mode lsgan --gpu_ids 0
```
Then, use Processing_image.ipynb to get the image for the pose estimation model training.


Pose Estimation Model Training:
```
python Pose_Model_Train.py
```

Hybrid Dataset Construction:
```
python create_hybrid_data.py
```

Pose Estimation Model Training Using Hybrid Dataset:
```
python Pose_model_Train_Hybrid.py
```
