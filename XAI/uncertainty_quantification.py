import torch
from whale_call_project.models.CNN import CNN
from torch.utils.data import DataLoader
from whale_call_project.data_loading.audio_dataloader import AudioDataset
from torchmetrics.classification import CalibrationError
import torch.nn.functional as F


def predict_with_MC_dropout(inputs):
    model = CNN()
    model.load_state_dict(torch.load("models/CNNmodel.pth"))
    model.eval()

    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    preds = []
    with torch.no_grad():
        for _ in range(30):
            output = model(inputs)  # logits or probs
            preds.append(torch.nn.functional.softmax(output, dim=1))  # or just output if you want logits

    # Stack and average
    stacked = torch.stack(preds)  # [T, B, C]
    mean_probs = stacked.mean(dim=0)  # [B, C]
    return mean_probs, stacked.std(dim=0)  # (mean, uncertainty)


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
