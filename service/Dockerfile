FROM python:3.8

WORKDIR /app

COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -U kolibri-ml
RUN pip install -U kdmt>=1.61
COPY . .

EXPOSE 8000

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["python", "api.py"]