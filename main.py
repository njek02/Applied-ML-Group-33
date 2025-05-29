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
    title="Whale Sound Detection API",
    summary="API for detecting whale sounds in audio files.",
    description="This API allows users to upload audio files (.aiff files) and receive predictions on whether whale sounds are present in the audio file.",
    )

@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url='/docs')


@app.post("/predictions", description="Predict if whale sounds are present in the uploaded audio file (.aiff).")
async def predict(audio_file: UploadFile = File(...)):

    if not audio_file.filename.endswith('.aiff'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .aiff files are accepted.")
    
    preprocessed_sample = preprocess_sample(audio_file.file)
    if preprocessed_sample is None:
        raise HTTPException(status_code=400, detail="Error processing the audio file.")
    
    preprocessed_sample = torch.tensor(preprocessed_sample, dtype=torch.float32)
    preprocessed_sample = preprocessed_sample.unsqueeze(0)

    model = CNN()
    model.load_state_dict(torch.load("CNNmodel.pth"))
    model.eval()
    
    prediction = model(preprocessed_sample)
    _, predicted_class = torch.max(prediction, 1)
    
    if predicted_class.item() == 0:
        predicted_label = "There is no whale sound"
    else:
        predicted_label = "There is a whale sound"
    return Prediction(
        filename = audio_file.filename,
        prediction= predicted_label
    )

