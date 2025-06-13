from whale_call_project.models.CNN import CNN
from whale_call_project.data_loading.audio_dataloader import AudioDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import random


if __name__ == "__main__":
    # Set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Load Datasets
    training_set = AudioDataset("data/training_data/", "data/data_labels/training_data.csv")
    print("Training Dataset Loaded")
    validation_set = AudioDataset("data/validation_data/", "data/data_labels/validation_data.csv")
    print("Validation Dataset Loaded")
    test_set = AudioDataset("data/test_data/", "data/data_labels/test_data.csv")
    print("Test Dataset Loaded")

    # Convert Datasets into DataLoaders
    train_loader = DataLoader(training_set, batch_size=128)
    print("Training DataLoader created")
    val_loader = DataLoader(validation_set, batch_size=128)
    print("Validation DataLoader created")
    test_loader = DataLoader(test_set, batch_size=128)
    print("Test DataLoader created")

    class_weights = [0.65294447, 2.1345802]

    # Train CNN
    model = CNN(class_weights=class_weights)
    print("Training model...")
    model.train_model(train_loader, val_loader)
    print("Evaluating model...")
    model.evaluate_model(test_loader)


    # Save the Model
    while True:
        user_input = input("Do you want to save the model? (y/n): ")
        if user_input.lower() == 'y':
            print("Model saved")
            torch.save(model.state_dict(), "CNNmodel.pth")
            break
        elif user_input.lower() == 'n':
            print("Model not saved.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")