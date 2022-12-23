FROM python:3.8

WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -U mlflow

COPY . .

EXPOSE 8000

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["python", "api.py"]