import os
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import task, flow

from RNN_MUSIC_GENERATOR.Processing_NEW import processing #check how will be last arquitecture
from registry import mlflow_transition_model
from params import *

@task
#how is the process of preprocess the new data? check with cristhian
def preprocess_new_data():
    return processing()

@task
def re_train(min_date: str, max_date: str, split_ratio: str):
    new_perf = train(min_date=  min_date, max_date=max_date, split_ratio= split_ratio)
    return new_perf

@task
def transition_model(current_stage: str, new_stage: str):
    return mlflow_transition_model(current_stage= current_stage, new_stage= new_stage)

@task
def notify(old_mae, new_mae):
    """
    Notify about the performance
    """
    base_url = 'https://wagon-chat.herokuapp.com'
    channel = '1257'
    url = f"{base_url}/{channel}/messages"
    author = 'arielimaa'

    if new_mae < old_mae and new_mae < 2.5:
        content = f"🚀 New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}"
    elif old_mae < 2.5:
        content = f"✅ Old model still good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
    else:
        content = f"🚨 No model good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
    data = dict(author=author, content=content)
    response = requests.post(url, data=data)
    response.raise_for_status()


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    Build the Prefect workflow for the `RNN_MUSIC_GENERATOR package. It should:
        - preprocess new data.
        - compute `new_mae` by re-training the current production model on this new data.
        - if the new one is better than the a old one, replace the current production model with the new one.
    """

    # Define the orchestration graph ("DAG")
    preprocessed = preprocess_new_data.submit(min_date=min_date, max_date=max_date)
    old_mae = evaluate_production_model.submit(min_date=min_date, max_date=max_date, wait_for=[preprocessed])
    new_mae = re_train.submit(min_date=min_date, max_date=max_date, split_ratio = 0.2, wait_for=[preprocessed])

    # Compute results as actual python object
    old_mae = old_mae.result()
    new_mae = new_mae.result()

    # Compare results
    if new_mae < old_mae:
        print("New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}")
        transition_model.submit(current_stage="Staging", new_stage="Production")
    else:
        print(f"Old model kept in place with MAE: {old_mae}. The new MAE was: {new_mae}")

    notify.submit(old_mae, new_mae)
train_flow()

if __name__ == "__main__":
    train_flow()
