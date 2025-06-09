import torch
from whale_call_project.models.CNN import CNN
from torch.utils.data import DataLoader
from whale_call_project.data_loading.audio_dataloader import AudioDataset
from torchmetrics.classification import CalibrationError
import torch.nn.functional as F
from torch import Tensor


def predict_with_MC_dropout(inputs: Tensor) -> tuple[Tensor, Tensor]:
    """
    Perform prediction with Monte Carlo Dropout to estimate uncertainty.

    Args:
        inputs (Tensor): Input tensor of shape (B, C, H, W), where B is the batch size.

    Returns:
        tuple[Tensor, Tensor]:
            - mean_probs (Tensor): Mean class probabilities across stochastic forward passes, shape (B, num_classes).
            - uncertainty (Tensor): Standard deviation across forward passes, representing uncertainty, shape (B, num_classes).
    """
    model = CNN()
    model.load_state_dict(torch.load("models/CNNmodel.pth"))
    model.eval()

    # Enable dropout layers during inference
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    preds = []
    with torch.no_grad():
        for _ in range(30):
            output = model(inputs)
            preds.append(F.softmax(output, dim=1))

    # Stack and average predictions
    stacked = torch.stack(preds)  # shape: (30, B, num_classes)
    mean_probs = stacked.mean(dim=0)
    uncertainty = stacked.std(dim=0)

    return mean_probs, uncertainty


if __name__ == "__main__":
    # Load test set
    test_set = AudioDataset("data/test_data/", "data/data_labels/test_data.csv")

    # Convert Datasets into DataLoaders
    test_loader = DataLoader(test_set, batch_size=128)
    images, labels = next(iter(test_loader))

    mean_probs, uncertainty = predict_with_MC_dropout(images)

    ece = CalibrationError(num_classes=2, n_bins=15, norm='l1', task='multiclass')

    ece_score = ece(mean_probs, labels)
    print(f"ECE Score: {ece_score.item():.4f}")
