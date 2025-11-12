Flood Segmentation using U-Net

This project implements a U-Net architecture in PyTorch for flood detection and segmentation from satellite or aerial images. It takes paired images and binary masks as input and trains a model to predict flooded regions.
Model Overview

The project uses a U-Net architecture designed for semantic segmentation.

Encoder (contracting path): captures context via convolution and pooling

Decoder (expanding path): reconstructs fine details using upsampling and skip connections

Output: a single-channel mask where each pixel represents flood (1) or non-flood (0)

Components
1. FloodDataset

A custom PyTorch Dataset class that:

Loads .jpg images and corresponding .png masks

Applies resizing and tensor conversion

Converts mask to binary (0 = background, 1 = flood region)

Example:

dataset = FloodDataset(images_dir="path/to/Image", masks_dir="path/to/Mask", transform=transform)

2. UNet Model

Defines the encoder–decoder structure:

DoubleConv: two 3×3 conv layers with ReLU

Skip connections between encoder and decoder

Final sigmoid layer for binary segmentation mask

Example:

model = UNet(in_ch=3, out_ch=1)

3. Training Pipeline

Loss: Binary Cross Entropy (BCELoss)

Optimizer: Adam (learning rate = 0.001)

Input image size: 256×256

Batch size: 4

Epochs: 50

Device: GPU if available

Command to train:

python flood_segmentation.py


Example output:

Epoch [1/50] Loss: 0.4278
Epoch [2/50] Loss: 0.3192
...
Model trained and saved as flood_unet.pth

How to Run

Install dependencies:

pip install torch torchvision pillow


Set paths in the script:

images_dir = r"F:\sih25\archive (4)\Image"
masks_dir = r"F:\sih25\archive (4)\Mask"


Run training:

python flood_segmentation.py


The model will be saved automatically as:

flood_unet.pth

Output

After training, the model predicts flood regions from input images.
Example (visual representation):

Input Image	Ground Truth Mask	Predicted Mask
image.jpg	mask.png	predicted_mask.png
Next Steps

Add validation metrics (Dice, IoU)

Implement data augmentation

Visualize training loss using Matplotlib

Deploy using Streamlit for real-time image uploads and predictions

Author

Anjali Srivastava
B.Tech CSE (DS + AI)
Shri Ramswaroop Memorial University
Flood Detection using U-Net (PyTorch Implementation)
