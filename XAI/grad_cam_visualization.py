from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from whale_call_project.models.CNN import CNN
from whale_call_project.preprocessing.preprocess_CNN import preprocess_sample
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import streamlit as st


def visualize_grad_cam(model: CNN, target_layers: list[torch.nn.Module], file_path: str, label: int, use_st: bool = False) -> None:
    """
    Generate and display a Grad-CAM visualization, optionally using Streamlit.

    Args:
        model (CNN): The CNN model.
        target_layers (List[torch.nn.Module]): Layers to target for Grad-CAM.
        file_path (str): Path to the input image.
        label (int): Target class label.
        use_st (bool): Whether to display using Streamlit.
    """
    grayscale_cam, original_image = generate_heatmap(model, target_layers, file_path, label)

    if use_st:
        show_heatmap_streamlit(grayscale_cam, file_path, original_image)
    else:
        show_heatmap(grayscale_cam, file_path, original_image)


def generate_heatmap(model: CNN, target_layers: list[torch.nn.Module], file_path: str, label: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a Grad-CAM heatmap for a given image and model.

    Args:
        model (CNN): The CNN model.
        target_layers (List[torch.nn.Module]): Layers to target for Grad-CAM.
        file_path (str): Path to the input image.
        label (int): Target class label.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The heatmap and the original image.
    """
    model.eval()
    cam = GradCAM(model=model, target_layers=target_layers)

    input_image = preprocess_sample(file_path)
    input_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)

    targets = [ClassifierOutputTarget(label)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    return grayscale_cam, input_image


def show_heatmap(grayscale_cam: np.ndarray, file_path: str, input_image: np.ndarray) -> None:
    """
    Display the Grad-CAM and original image using Matplotlib.

    Args:
        grayscale_cam (np.ndarray): Grad-CAM heatmap.
        file_path (str): Path to the input image.
        input_image (np.ndarray): Original input image.
    """
    resized_cam = cv2.resize(grayscale_cam, (64, 64))
    rgb_image = preprocess_sample(file_path, rgb_output=True)
    original_image = cv2.resize(input_image.squeeze(0), (64, 64))

    grad_cam_visualization = show_cam_on_image(rgb_image, resized_cam, use_rgb=True)

    # Plot heatmap
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, origin='lower', cmap='viridis')
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(grad_cam_visualization, origin='lower')
    plt.axis('off')
    plt.title('Grad-CAM Visualization')

    plt.tight_layout()
    plt.show()


def show_heatmap_streamlit(grayscale_cam: np.ndarray, file_path: str, input_image: np.ndarray) -> None:
    """
    Display the Grad-CAM and original image side-by-side in Streamlit.

    Args:
        grayscale_cam (np.ndarray): Grad-CAM heatmap.
        file_path (str): Path to the input image.
        input_image (np.ndarray): Original input image.
    """
    resized_cam = cv2.resize(grayscale_cam, (64, 64))
    rgb_image = preprocess_sample(file_path, rgb_output=True)
    original_image = cv2.resize(input_image.squeeze(0), (64, 64))

    grad_cam_visualization = show_cam_on_image(rgb_image, resized_cam, use_rgb=True)

    # Flip vertically to simulate matplotlib origin='lower'
    original_image = np.flipud(original_image)
    grad_cam_visualization = np.flipud(grad_cam_visualization)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(grad_cam_visualization, caption="Grad-CAM Visualization", use_column_width=True)

if __name__ == "__main__":

    model = CNN()
    model.load_state_dict(torch.load("models/CNNmodel_class_w.pth"))

    file_path = "train168.aiff"

    target_layers = [model.model_layers[10]]

    visualize_grad_cam(model, target_layers, file_path, label=0)
