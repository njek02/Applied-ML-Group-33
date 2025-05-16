from old_stuff.Old_CNN import CNN
from project_name.data_loading.audio_dataloader import AudioDataset
from torch.utils.data import DataLoader
# from preprocessing.split_data import split_data_folder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

# Comment out if data split not done yet
# csv_path = "data\\train.csv"
# dataset_path = "data\\dataset"
# split_data_folder(csv_path, dataset_path)


training_set = AudioDataset("data/training_data/", "data/data_labels/training_data.csv")
print("Training Loaded")
validation_set = AudioDataset("data/validation_data/", "data/data_labels/validation_data.csv")
print("Validation Loaded")


# Compute class weights
df = pd.read_csv('data/data_labels/training_data.csv')

labels = df['label'].values

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=np.array(labels)
)
print("Computed class weights")


train_loader = DataLoader(training_set, batch_size=128)
print("Train DataLoader created")
val_loader = DataLoader(validation_set, batch_size=128)
print("Val DataLoader created")

train_loader.dataset

# Train CNN
model = CNN(class_weights)
model.train_model(train_loader, val_loader)
