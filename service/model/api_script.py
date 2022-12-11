# -*- coding: utf-8 -*-

from fastapi import FastAPI
import uvicorn
from pydantic import create_model
from kolibri.model_loader import ModelLoader
import os
import pandas as pd

# Create the app
app = FastAPI()

model_interpreter = ModelLoader.load("/app/model/current")

# Create input/output pydantic models
input_model = create_model("/app/model/api_script_input", **{'Text': '300 Boston Scientific Way I Marlborough, MA 01752'})
output_model = create_model("/app/model/api_script_output", None_prediction="X")


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = model_interpreter.predict(data)
    return {"None_prediction": predictions["Prediction"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)