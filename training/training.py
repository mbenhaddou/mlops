import os

import pandas as pd
from kolibri.config import ModelConfig
from kolibri.model_loader import ModelLoader
from kolibri.model_trainer import ModelTrainer
from prefect import task, flow
from prefect.tasks import task_input_hash
from datetime import timedelta
from kdmt.mlflow import get_stage_version, get_best_run, register_model


@task
def get_config(model_name, output_folder: str='./model', optimize=False, budget=3600, calibrate=False, register=False):
    """
    Test multi label classification
    """


    confg = {}

    confg['format'] = 'csv'
    confg['track-experiments']=True
    confg['register-model']=register
    confg['calibrate-model']=calibrate
    confg['experiment-uri']=os.getenv('MLFLOW_TRACKING_URI')
    confg['experiment-name']='email_signature'
    confg['remove-stopwords'] = True
    confg['model-name']=model_name
    confg['do-lower-case'] = True
    confg['max-time-for-optimization'] = budget

    confg['language']='en'
    confg["multi-label"] = False
    confg["evaluate-performance"] = True
    confg['max-features'] = 20000
    confg['output-folder'] = output_folder
    confg['pipeline']= ['WordTokenizer', 'TFIDFFeaturizer', 'SklearnEstimator']
    confg["model"] = 'LogisticRegression'
    confg['optimize-pipeline']=optimize
    confg['n-jobs']=-1
    confg['optimization-n-jobs']=1
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
def training(model_name, content_col="Text", target_col="Category"):
    data = load_data()
    config=get_config(model_name=model_name)
    X, y = transform_data(data, content_col, target_col)

    model = train_model(config, X, y)
    save_model(model, "model")
@flow(name="better-model")
def seach_better_models(model_name, content_col="Text", target_col="Category", metric="F1"):

    data = load_data()
    config=get_config(model_name=model_name,optimize=True, budget=1800)
    X, y = transform_data(data, content_col, target_col)

    model = train_model(config, X, y)



@flow(name="select_model")
def find_model_condidate():
    tolerance = 0.01

    prod_version = get_stage_version("email_signature")
    best_run=get_best_run("email_signature", "F1")

    is_after_prod=best_run['start_time']>prod_version['creation_timestamp']

    if is_after_prod and prod_version['metrics']['F1'] < best_run['metrics']['F1']*(1-tolerance):
        print('Found Better model')
        artifact=prod_version["source"].split('/')[-1]
        register_model("email_signature", artifact, stage="Production")

@flow(name="prediction")
def prediction(script_path):
    lanuch_api(script_path)


if __name__ == "__main__":
    os.environ['MLFLOW_TRACKING_URI']="http://0.0.0.0:5000"
#    prediction("model/api_script.py")
    training('email_signature')
