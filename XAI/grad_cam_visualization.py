from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from whale_call_project.models.CNN import CNN
from whale_call_project.preprocessing.preprocess_CNN import preprocess_sample
import torch
import matplotlib.pyplot as plt


model = CNN()
model.load_state_dict(torch.load("models/CNNmodel.pth"))
model.eval()

input_image = preprocess_sample("data/training_data/train168.aiff")
input_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)

rgb_image = preprocess_sample("data/training_data/train168.aiff", rgb_output=True)

target_layers = [model.model_layers[10]]

cam = GradCAM(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(1)]

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

# Plot heatmap
plt.figure(figsize=(6, 6))
plt.imshow(rgb_image, origin='lower')
plt.axis('off')
plt.title('Grad-CAM Visualization')
plt.show()