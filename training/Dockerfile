FROM python:3.8

RUN mkdir /training
WORKDIR /training

COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -U kolibri-ml
COPY . .

EXPOSE 4200

#RUN prefect config set PREFECT_API_URL=0.0.0.0:4200/api
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["sh","-c","prefect orion start --host 0.0.0.0 & sleep 25 && prefect deployment apply training-deployment.yaml & sleep 15 && prefect agent start -q 'ml'"]
