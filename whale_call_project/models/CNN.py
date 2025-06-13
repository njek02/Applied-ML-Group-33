import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, ConfusionMatrixDisplay, PrecisionRecallDisplay


class CNN(nn.Module):
    """A Convolutional Neural Network (CNN) model for image classification."""
    def __init__(self, class_weights: np.ndarray = None) -> None:
        """
        Initializes the CNN model by defining the layers.
        """
        super().__init__()
        self.class_weights = class_weights

        self.model_layers = nn.Sequential(
            # Convolutional block 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),  # (8, 32, 32)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (8, 16, 16)
            nn.Dropout(p=0.3),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),  # (16, 16, 16)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 8, 8)
            nn.Dropout(p=0.3),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # (32, 8, 8)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 4, 4)
            nn.Dropout(p=0.3),

            nn.Flatten(),  # Flatten the Tensor into a 1D output

            # Fully connected layers
            nn.Linear(in_features=32 * 4 * 4, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
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
        self.to(device)
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, dtype=torch.float).to(device)
        else:
            weights = None

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(params=self.parameters(), lr=0.001, weight_decay=0.0001)

        num_epochs = 15
        train_losses, validation_losses = [], []

        for epoch in range(num_epochs):
            self.train()
            # Training phase
            train_loss = 0
            for inputs, labels in training_data:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            self.eval()

            predictions = []
            ground_truths = []
            probabilities = []


            with torch.no_grad():
                for inputs, labels in test_data:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self(inputs)
                    probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1

                    predicted = (probs >= 0.48)

                    probabilities.append(probs)
                    predictions.append(predicted)
                    ground_truths.append(labels)

                predictions = torch.cat(predictions).cpu().numpy()
                ground_truths = torch.cat(ground_truths).cpu().numpy()
                probabilities = torch.cat(probabilities).cpu().numpy()

            # Metrics
            # F-1 score
            f_1_score = f1_score(y_true=ground_truths, y_pred=predictions, average="binary")

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_true=ground_truths, y_pred=predictions)

            # Area under ROC
            auroc = roc_auc_score(y_true=ground_truths, y_score=probabilities)

            # Accuracy
            accuracy = accuracy_score(y_true=ground_truths, y_pred=predictions)

            # Results
            print(f"F-1 Score: {f_1_score:.4f}")
            print(f"Area under ROC: {auroc:.4f}")
            print(f"Accuracy: {accuracy:.4f}")

            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(ground_truths))
            disp.plot(cmap='Blues', values_format='d')

            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.show()

            # Precision Recall curve

            display = PrecisionRecallDisplay.from_predictions(
            ground_truths, probabilities, name="CNN", plot_chance_level=True, despine=True
            )
            _ = display.ax_.set_title("Precision-Recall curve")

            plt.show()

