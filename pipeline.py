from old_stuff.Old_CNN import CNN
from project_name.data.audio_dataloader import AudioDataset
from torch.utils.data import DataLoader
from preprocessing.split_data import split_data_folder


# Comment out if data split not done yet
# csv_path = "data\\train.csv"
# dataset_path = "data\\dataset"
# split_data_folder(csv_path, dataset_path)


training_set = AudioDataset("data/training_data/", "data/data_labels/training_data.csv")
print("Training Loaded")
validation_set = AudioDataset("data/validation_data/", "data/data_labels/validation_data.csv")
print("Validation Loaded")

train_loader = DataLoader(training_set, batch_size=128)
print("Also done")
val_loader = DataLoader(validation_set, batch_size=128)
print("Time to start training")


# Train CNN
model = CNN()
model.train_model(train_loader, val_loader)
