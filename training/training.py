import cProfile
import os
import mlflow

import pandas as pd
from kolibri.config import ModelConfig
from kolibri.model_loader import ModelLoader
from kolibri.model_trainer import ModelTrainer
from prefect import task, flow
from prefect.tasks import task_input_hash
from datetime import timedelta



@task
def get_config(output_folder: str='./model'):
    """
    Test multi label classification
    """


    confg = {}

    confg['format'] = 'csv'
    confg['track-experiments']=True
    confg['register-model']=True
    confg['calibrate-model']=True
    confg['experiment-uri']=os.getenv('MLFLOW_TRACKING_URI')
    confg['experiment-name']='email_signature'
    confg['remove-stopwords'] = True
    confg['language'] = 'fr'
    confg['do-lower-case'] = True

 #   confg["sequence_length"] = 60
    confg['language']='fr'
    confg['log-plots']=[]#["pr", "confusion_matrix", "roc", "errors",  "class_report", "calibration"]
    confg["multi-label"] = False
    confg["evaluate-performance"] = True
    confg['max-features'] = 20000
    confg['output-folder'] = output_folder
    confg['pipeline']= ['WordTokenizer', 'TFIDFFeaturizer', 'SklearnEstimator']
    confg["model"] = 'LogisticRegression'
#    confg['pipeline'] = ['DnnTextClassEstimator']
#    confg["model"] = 'cnn_attention'

    confg['n-jobs']=-1

    return confg
@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def load_data():
    corpus = pd.read_excel("https://www.dropbox.com/scl/fi/2wtas19x68m57sr8e1wz3/email_data.xlsx?dl=1&rlkey=hdcc643wc6li33bpf4yu7pyvz")
    return corpus
@task
def transform_data(corpus, content_col: str ='Text', target_col: str ='Category'):
    corpus=corpus.dropna(subset=[content_col]).sample(n=30000)


    X = corpus[content_col].values.tolist()
    y = corpus[target_col].values.tolist()

    return X, y

@task
def train_model(configs, X, y):

    trainer=ModelTrainer(ModelConfig(configs))
    data=pd.DataFrame(X)
    data.columns=["Text"]
    trainer.fit(data, y)
    return trainer
@task
def save_model(trainer, save_path):

    model_directory = trainer.persist(save_path, fixed_model_name="current")
@task
def load_model(save_path):
    model_interpreter = ModelLoader.load(
        os.path.join(save_path, 'current'))
    return model_interpreter

@task
def create_api(model_interpreter, path):
    model_interpreter.create_api(path)

@task
def predict(model_interpreter, text):
    return  model_interpreter.predict(text)


@task
def lanuch_api(path):
    script=open(path, "r").read()
    exec(script)

@flow(name="modeling")
def training(content_col="Text", target_col="Category", save_path="./model"):
    data = load_data()
    config=get_config()
    X, y = transform_data(data, content_col, target_col)

    model = train_model(config, X, y)

    save_model(model, save_path)

    model_interpreter=load_model(save_path)
    create_api(model_interpreter, os.path.join(save_path,"api_script"))

@flow(name="prediction")
def prediction(script_path):
    lanuch_api(script_path)


if __name__ == "__main__":
#    prediction("model/api_script.py")
    training()
