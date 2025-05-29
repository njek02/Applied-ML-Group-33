# Right Whale Upcall Classifier
This is the GitHub page for a Right Whale Upcall Classifier using a PyTorch convolutional neural network. The data is sourced from https://www.kaggle.com/competitions/whale-detection-challenge/data 

## Dependencies
To install the required packages, run pipenv install to load the required dependencies from the Pipfile and Pipfile.lock files.
If you are using a different package manager that is compatible with pipenv, see the documentation of your package manager on how to load dependencies from a Pipfile.

## Model Instructions
To use our model, the following steps need to be followed:

### 1. Download the dataset
Go to https://www.kaggle.com/competitions/whale-detection-challenge/data and download "whale_data.zip". Since this dataset was part of a contest, a Kaggle account is required.

### 2. Extract the .zip file
Extract the .zip file to your desired destination. It is recommended to extract it to the "data" folder found in the root directory. 
Ensure the extracted folder contains a folder called train that contains .aiff files called "trainx.aiff", where x is a number.
Furthermore, ensure that the extracted folder contains a file containing the labels called train.csv.

### 3. Split the dataset into a training, validation and test set
Run setup.py to automatically populate the "data" folder with a training, validation and test split.
This will create a "data_labels" folder with three .csv files for the labels and three folders that will contain the .aiff samples.

### 4. Run a pipeline
Now you can train our SVM baseline or our CNN model by running their respective pipelines.
Make sure the root directory is your working directory and run files as a module. For example: "python -m pipelines.pipeline_CNN"


## API Instructions
To run our API, the following steps need to be followed:

### 1. Ensure all dependencies are installed
If you have not done so already, install the required dependencies. Instructions can be at the start.
### 2. Run the API application
Run "uvicorn main:app" in a terminal to launch the API.
The API is now running locally on http://127.0.0.1:8000 or an equivalent localhost IP adress.
### 3. Upload a file to try out the API
To try out the API, go to the "predict" section and click on "try it out". 
Now you can upload a file to receive a prediction from our model.
