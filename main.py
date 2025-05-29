from fastapi import FastAPI, UploadFile, HTTPException, File
from whale_call_project.preprocessing.preprocess_CNN_sample import preprocess_sample
from whale_call_project.models.CNN import CNN
import torch
from starlette.responses import RedirectResponse
from pydantic import BaseModel

class Prediction(BaseModel):
    """
    Class for the prediction response model.
    """    
    filename: str
    prediction: str

app = FastAPI(
    title="Right Whale Upcall Classifier",
    summary="An API endpoint to classify Right whale upcalls using a CNN. Trained on spectrograms.",
    description="""
# An API endpoint to access a CNN trained on spectrograms.
## Model usage
The model is trained on 32x32 mel-spectrograms of Right whale upcalls, random noise or other whale calls. \n
It is designed to receive 2 second long .aiff audio fragments at a sampling rate of 2000 Hz.

## Limitations
Converting the raw audio into spectrograms is tuned to 2 second audio fragments with a sampling rate of 2000 Hz. \n
Thus, audio fragments with a different sampling rate will result in low quality spectrograms and will likely lead to wrong predictions by the model. \n
Similarly, .wav files result in unusable spectrograms for our model. Thus only .aiff files are accepted. \n
\n
If the audio fragment is longer than 2 seconds, the remaining audio will not be converted into a spectrogram. \n
This means that any relevant audio must be in the first two seconds of the audio fragment. \n
\n
If the audio fragment is shorter than 2 seconds, the resulting spectrogram will cause the model to give incorrect predictions. \n
Thus only audio fragments of 2 seconds or longer are accepted.
    """,
)

@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url='/docs')


@app.post("/predictions", description=("Right whale upcall classifier endpoint. "
                                      "Audio file should be a .aiff file with a sampling rate of 2000 Hz. "
                                      "Furthermore, all relevant information should be in the first 2 seconds. "
                                      "Returns the prediction as a string."))
async def predict(audio_file: UploadFile = File(...)):

    if not audio_file.filename.endswith('.aiff'):
        raise HTTPException(status_code=415, detail="Invalid file type. Only .aiff files are accepted.")
    
    # Preprocess sample
    try:
        preprocessed_sample = preprocess_sample(audio_file.file)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    
    preprocessed_sample = torch.tensor(preprocessed_sample, dtype=torch.float32)
    preprocessed_sample = preprocessed_sample.unsqueeze(0)

    # Load model
    model = CNN()
    model.load_state_dict(torch.load("CNNmodel.pth"))
    model.eval()
    
    # Make a prediction
    prediction = model(preprocessed_sample)
    _, predicted_class = torch.max(prediction, 1)
    
    # Convert prediction to labek
    if predicted_class.item() == 0:
        predicted_label = "There is no Right whale upcall present in this fragment"
    else:
        predicted_label = "There is a Right whale upcall present in this fragment"
    return Prediction(
        filename = audio_file.filename,
        prediction= predicted_label
    )

