FROM jupyter/minimal-notebook:python-3.8

WORKDIR /notebooks/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER root
RUN ldconfig && \
    jupyter nbextension enable toc2/main --sys-prefix && \
    jupyter nbextension enable collapsible_headings/main --sys-prefix

ENV MLFLOW_TRACKING_URI=http://mlflow:5000