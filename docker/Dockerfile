FROM continuumio/miniconda3:latest

ARG PYTHON_VERSION=3.8.8

RUN apt-get update \
    && apt-get install --yes \
        libglu1-mesa-dev \
        freeglut3-dev \
        mesa-common-dev \
        vim \
        graphviz \
        git \
    && apt-get clean 

RUN conda install --yes --quiet -c conda-forge -c anaconda \
    python=$PYTHON_VERSION \
    root=$ROOT_VERSION \
    numpy \
    pandas \
    matplotlib \ 
    uproot \
    keras \
    scikit-learn \
    scikit-image \
    tensorflow-gpu \
    && conda clean --yes --all --force-pkgs-dirs 

RUN pip install --quiet snakemake

RUN pip install --quiet scikinC
      
WORKDIR ${JUPYTER_ROOT}


