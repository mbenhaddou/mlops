# -*- coding: utf-8 -*-

from fastapi import FastAPI
import mlflow
import uvicorn
from pydantic import create_model
from kolibri.model_loader import ModelLoader
from kdmt.mlflow import download_model, is_model_registered_by_name, get_stage_version
import os
import pandas as pd

# Create the app
app = FastAPI()

model = is_model_registered_by_name("email_signature")
print("Model is registered: ", model)
prduction={}
if model:
    prduction = get_stage_version("email_signature")
    print(prduction)
try:
    model_interpreter=mlflow.pyfunc.load_model(model_uri=prduction["source"], dst_path="/app/model")

except Exception as e:
    pass

# Create input/output pydantic models
input_model = create_model("/app/model/api_script_input", **{'Text': '300 Boston Scientific Way I Marlborough, MA 01752'})
output_model = create_model("/app/model/api_script_output", None_prediction="X")



# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    print(data)
    data = pd.DataFrame([data.dict()])
    predictions = model_interpreter.predict(data)
    return {"None_prediction": predictions["Prediction"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)