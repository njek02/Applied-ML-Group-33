import streamlit as st
from whale_call_project.models.CNN import CNN
from whale_call_project.models.SVM import SVCModel
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from whale_call_project.preprocessing.preprocess_SVM import Preprocess
import pandas as pd
import torch
from whale_call_project.preprocessing.wave_to_image import wave_to_spec
from whale_call_project.preprocessing.normalization import spectrogram_normalization
from sklearn.decomposition import PCA
from whale_call_project.preprocessing.preprocess_CNN import preprocess_sample
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import soundfile as sf


st.title("Model prediction on whale sounds")

model = st.selectbox("Select model", ["CNN", "SVM"])
upload_file = st.file_uploader("Upload an AIFF file", type= ["aiff"])


if upload_file is not None:
    if not upload_file.name.endswith(".aiff"):
        st.error("Please upload a valid AIFF file.")
    else:
        st.success("File uploaded successfully!")

        test_file_path = r"C:\Users\tomas\OneDrive\Desktop\Second year\Applied ML\Applied-ML-Group-33\train168.aiff"
        st.success(f"Using test file: {test_file_path}")

        if st.button("RUN MODEL"):
            if model == "CNN":
                try:
                    # Let preprocess_sample handle any resampling
                    spec_image = preprocess_sample(test_file_path, rgb_output=False)
                    st.write("Spectrogram image (CNN input):")
                    fig, ax = plt.subplots()
                    librosa.display.specshow(spec_image.squeeze(), x_axis='time', y_axis='mel', ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)

                    # Prepare input for CNN (add batch and channel dimensions)
                    input_tensor = torch.tensor(spec_image, dtype=torch.float32).unsqueeze(0)  # (1, 1, H, W)

                    # Load trained CNN model
                    model = CNN()
                    model.load_state_dict(torch.load("models/CNNmodel.pth", map_location=torch.device('cpu')))
                    model.eval()

                    # Predict
                    with torch.no_grad():
                        output = model(input_tensor)
                        predicted_class = torch.argmax(output, dim=1).item()
                    st.write(f"Predicted class: {predicted_class}")

                except Exception as e:
                    st.error(f"Error during preprocessing or prediction: {e}")



            if model == "SVM":
                """
                features = create_spectrogram_from_filelike(upload_file)
                training_data = np.load("data/x_train.npy")
                train_dataset = pd.read_csv("data/training_data.csv")

                # Debug: Show feature shapes
                st.write(f"Shape of features from uploaded file: {features.shape}")
                st.write(f"Shape of training data: {training_data.shape}")

                custom_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=0.95)),  # or n_components=features.shape[1]
                    ('svm', SVCModel())
                ])

                custom_pipeline.fit(training_data, train_dataset["label"])
                prediction = custom_pipeline.predict(features)
                st.write(f"Prediction SVM: {prediction[0]}")
                """
                pass