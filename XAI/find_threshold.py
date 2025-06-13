from whale_call_project.models.CNN import CNN
from whale_call_project.data_loading.audio_dataloader import AudioDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import torch

validation_set = AudioDataset("data/validation_data/", "data/data_labels/validation_data.csv")  
val_loader = DataLoader(validation_set, batch_size=128)


model = CNN()
model.load_state_dict(torch.load("models/CNNmodel.pth"))

model.eval()
probs = []
targets = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs
        labels = labels

        logits = model(inputs)  # shape: [batch_size, 2]
        softmax_probs = torch.softmax(logits, dim=1)  # shape: [batch_size, 2]

        # Class 1 probabilities
        class1_probs = softmax_probs[:, 1]  # shape: [batch_size]

        probs.extend(class1_probs.cpu().numpy())
        targets.extend(labels.cpu().numpy())

probs = np.array(probs)
y_true = np.array(targets)


best_f1 = 0
best_threshold = 0.5

for t in np.arange(0.0, 1.0, 0.01):
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_true, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold: {best_threshold}, Best F1: {best_f1}")