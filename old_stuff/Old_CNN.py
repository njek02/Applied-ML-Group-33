
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy, multiclass_confusion_matrix
import matplotlib.pyplot as plt


class CNN(nn.Module):
    """A Convolutional Neural Network (CNN) model for image classification."""
    def __init__(self) -> None:
        """
        Initializes the CNN model by defining the layers.
        """
        super().__init__()

        self.model_layers = nn.Sequential(
            # Convolutional block 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),  # (8, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (8, 32, 32)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),  # (16, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 8, 8)

            nn.Flatten(),  # Flatten the Tensor into a 1D output

            # Fully connected layers
            nn.Linear(in_features=16 * 8 * 8, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Activates the forward pass of the model given in the constructor.

        Input: x (torch.Tensor): Input tensor containing the batch of images.

        Returns:
            Output tensor after passing through the model layers.
        """
        return self.model_layers(x)


    def train_model(self, training_data: DataLoader, validation_data: DataLoader) -> None:
        """
        Trains the CNN model using the provided training and validation data.

        Input:
            training_data: DataLoader containing the training data.
            validation_data: DataLoader containing the validation data.
        Returns:
            None
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.parameters(), lr=0.01)

        num_epochs = 10
        train_losses, validation_losses = [], []

        for epoch in range(num_epochs):
            self.train()
            # Training phase
            train_loss = 0
            i = 0
            for inputs, labels in training_data:
                print("Step: ", i)
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                i += 1

            train_losses.append(train_loss / len(training_data))

            # Validation phase
            self.eval()
            validation_loss = 0
            with torch.no_grad():
                for inputs, labels in validation_data:

                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self(inputs)
                    _, predicted = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()

            validation_losses.append(validation_loss / len(validation_data))

            print(f"Epoch [{epoch+1}/{num_epochs}], " +
                  f"Train Loss: {train_losses[-1]:.4f}, " +
                  f"Validation Loss: {validation_losses[-1]:.4f}, ")

        print("Training finished")
        # Plot training and validation losses
        plt.plot(train_losses, label='Train Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.legend()
        plt.show()

        return


    def evaluate_model(self, test_data: DataLoader) -> None:
        """
        Evaluates the CNN model using the provided test data.

        Input:
            test_data: DataLoader containing the test data.

        Returns:
            None
        """
        predictions = torch.tensor([])
        ground_truths = torch.tensor([])

        with torch.no_grad():
            for inputs, labels in test_data:

                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)

                predictions = torch.cat((predictions, predicted))
                ground_truths = torch.cat((ground_truths, labels))

        # Metrics
        # Accuracy
        accuracy = multiclass_accuracy(predictions, ground_truths)

        # Log Loss (on final batch)
        log_loss = torch.nn.functional.cross_entropy(outputs, labels)

        # F-1 score
        f_1_score = multiclass_f1_score(predictions.to(dtype=torch.int64), 
                                        ground_truths.to(dtype=torch.int64),
                                        average='macro', num_classes=10)

        # Confusion Matrix
        confusion_matrix = multiclass_confusion_matrix(predictions.to(dtype=torch.int64),
                                                       ground_truths.to(dtype=torch.int64),
                                                       num_classes=10)

        # Results
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Log Loss: {log_loss:.4f}")
        print(f"F-1 Score: {f_1_score:.4f}")
        print(f"Confusion Matrix: {confusion_matrix}")
