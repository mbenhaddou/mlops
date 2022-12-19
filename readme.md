# MLOPS system with MLFLOW / PREFECT / Streamlit / Docker Compose

This project showcases the use of `FastAPI` and `Streamlit` in tandem.
The trained model is deployed using a FastAPI rest service containerized using `Docker`.
The front end UI is built on Streamlit which is hosted on its own Docker container.

We spin both the containers together using `Docker Compose`.

## setup Prefect server

Create a prefect deployment for the training server.
 
`prefect deployment build training.py:training -n train-classifier -q ml`


## How to use

Clone this repo and run the below docker command:

`To Start Application:`
```docker
docker-compose up -d --build
```
and navigate to http://localhost:8501/

`To Stop Application:`
```docker
docker-compose down
```

### Trivia: 

_Using the volume tag in the compose file, we can mount the local folders from your computer to the Docker container. Now you can develop your app while using Docker and save changes._