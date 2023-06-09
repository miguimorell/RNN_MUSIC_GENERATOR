import numpy as np
from RNN_MUSIC_GENERATOR.registry import load_model
from RNN_MUSIC_GENERATOR.Processing_NEW.processing import process_data #check the processing function

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.state.model = load_model(model_origin='mlflow')

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict():
    """
    Make a single bass song prediction.

    """
    model = app.state.model
    X_pred= #streamlit data
    X_processed_pred = process_data(X_pred) #ADD OUR FUNCTION
    y_pred = model.predict(X_processed_pred) #ADD

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return dict('song'=(y_pred))


@app.get("/")
def root():
    return {
    'greeting': 'Hello'
}
