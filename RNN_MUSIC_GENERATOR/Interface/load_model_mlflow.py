import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from params import *
#from Model.model_train_version2 import init_model, train_model
from RNN_MUSIC_GENERATOR.registry import load_model,save_model,save_results
from RNN_MUSIC_GENERATOR.registry import mlflow_run


@mlflow_run
def upload_model():

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """
    model = load_model(model_origin= 'local')

    # Save results on the hard drive using registry
    #save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive
    save_model(model=model)


    print("✅ upload() done \n")

    return None


# def pred(X_pred: pd.DataFrame = None): #check our format
#     """
#     Make a prediction using the latest trained model
#     """

#     print("\n⭐️ Use case: predict")

#     # if X_pred is None:
#     #     X_pred = pd.DataFrame(dict(
#     #     pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
#     #     pickup_longitude=[-73.950655],
#     #     pickup_latitude=[40.783282],
#     #     dropoff_longitude=[-73.984365],
#     #     dropoff_latitude=[40.769802],
#     #     passenger_count=[1],
#     # ))

#     model = load_model(model_origin='mlflow')
#     assert model is not None

#     X_processed = preprocess_features(X_pred) #ADD OUR FUNCTION
#     y_pred = model.predict(X_processed) #ADD

#     print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
#     return y_pred


if __name__ == '__main__':
    #preprocess(min_date='2009-01-01', max_date='2015-01-01') #our function to generate X_proc
    upload_model()
