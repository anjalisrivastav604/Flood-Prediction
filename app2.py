# app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Model
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = DoubleConv(256+512, 256)
        self.dconv_up2 = DoubleConv(128+256, 128)
        self.dconv_up1 = DoubleConv(128+64, 64)
        self.conv_last = nn.Conv2d(64, out_ch, 1)
    
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(self.maxpool(conv1))
        conv3 = self.dconv_down3(self.maxpool(conv2))
        conv4 = self.dconv_down4(self.maxpool(conv3))
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return torch.sigmoid(out)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model.load_state_dict(torch.load("flood_unet.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Streamlit UI
st.markdown("""
<div style="background-color: tomato; padding: 10px; border-radius:5px">
<h2 style="color:white; text-align:center;">Percentage Flooded Area</h2>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a flood image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor).squeeze().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.uint8)
        percentage = pred_binary.sum() / pred_binary.size * 100

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Flooded Area: **{percentage:.2f}%**")

    if percentage > 50:
        st.error("⚠ Hazard Predicted! Flooded area exceeds 50%")
    else:
        st.success("✅ Safe: Flooded area below hazard threshold")
