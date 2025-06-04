from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from whale_call_project.models.CNN import CNN
from whale_call_project.preprocessing.preprocess_CNN import preprocess_sample
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


model = CNN()
model.load_state_dict(torch.load("models/CNNmodel.pth"))
model.eval()

file_path = "data/training_data/train26822.aiff"

input_image = preprocess_sample(file_path)
input_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)

rgb_image = preprocess_sample(file_path, rgb_output=True)

target_layers = [model.model_layers[10]]

cam = GradCAM(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(1)]

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

grayscale_cam = grayscale_cam[0, :]
resized_cam = cv2.resize(grayscale_cam, (64, 64))

rgb_image = preprocess_sample(file_path, rgb_output=True)
visualization = show_cam_on_image(rgb_image, resized_cam, use_rgb=True)


original_image = cv2.resize(input_image.squeeze(0), (64, 64))

# Plot heatmap
plt.figure(figsize=(8, 4))

# First image (Normal Image)
plt.subplot(1, 2, 1)
plt.imshow(original_image, origin='lower', cmap='viridis')
plt.axis('off')
plt.title('Original Image')

# Second image (Grad-CAM)
plt.subplot(1, 2, 2)
plt.imshow(visualization, origin='lower')
plt.axis('off')
plt.title('Grad-CAM Visualization')

plt.tight_layout()
plt.show()
