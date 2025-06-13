import streamlit as st
from whale_call_project.models.CNN import CNN
import matplotlib.pyplot as plt
import librosa.display
import torch
from whale_call_project.preprocessing.preprocess_CNN import preprocess_sample
from XAI.grad_cam_visualization import visualize_grad_cam
import os

upload_folder = "uploads"
os.makedirs(upload_folder, exist_ok=True)

st.title("Model prediction on whale sounds")

model = st.selectbox("Select model", ["CNN"])
upload_file = st.file_uploader("Upload an AIFF file", type= ["aiff"])


if upload_file is not None:
    if not upload_file.name.endswith(".aiff"):
        st.error("Please upload a valid AIFF file.")
    else:
        file_path = os.path.join(upload_folder, upload_file.name)
        with open(file_path, "wb") as f:
            f.write(upload_file.getbuffer())

        st.success(f"Using file: {upload_file.name}")

        if st.button("RUN MODEL"):
            if model == "CNN":
                try:
                    # Let preprocess_sample handle any resampling
                    input_image = preprocess_sample(file_path, rgb_output=False)
                    
                    y, sr = librosa.load(file_path, sr=None)

                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=256, hop_length=64, n_mels=64, fmin=100, fmax=400)
                    db_mel_spec = librosa.power_to_db(mel_spec)


                    st.markdown("""
                        <div style='
                            margin-top: 30px;
                            margin-bottom: 10px;
                            padding: 10px 15px;
                            background-color: #e8f4fc;
                            color: #2c3e50;
                            font-size: 20px;
                            font-weight: 600;
                            border-left: 6px solid #1abc9c;
                            border-radius: 5px;
                        '>
                            📊 Spectrogram Image (CNN Input)
                        </div>
                    """, unsafe_allow_html=True)

                    fig, ax = plt.subplots()
                    librosa.display.specshow(db_mel_spec, x_axis='time', y_axis='mel', ax=ax, n_fft=256, hop_length=64, fmin=100, fmax=400)
                    st.pyplot(fig)
                    plt.close(fig)

                    # Prepare input for CNN (add batch and channel dimensions)
                    input_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)  # (1, 1, H, W)

                    # Load trained CNN model
                    model = CNN()
                    model.load_state_dict(torch.load("models/CNNmodel.pth"))
                    model.eval()

                    # Predict
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)[:, 1].item()  # Probability of class 1

                        predicted_class = 1 if (probs >= 0.48) else  0

                        if predicted_class == 1:
                            prediction_msg = "There is a Right whale upcall present."
                        else:
                            prediction_msg = "There is no Right whale upcall present."

                        st.markdown(f"""
                            <div style='
                                background-color:#f0f2f6;
                                padding:20px;
                                border-radius:10px;
                                text-align:center;
                                font-size:24px;
                                font-weight:bold;
                                color:#2c3e50;
                            '>
                                🐋 Prediction: <span style='color:#1abc9c'>{prediction_msg}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Plot justification heatmap
                    target_layers = [model.model_layers[10]]
                    visualize_grad_cam(model, target_layers, file_path=file_path, label=predicted_class, use_st=True)


                except Exception as e:
                    st.error(f"Error during preprocessing or prediction: {e}")
