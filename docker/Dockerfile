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
        build-essential \
        mono-mcs \
    && apt-get clean 


RUN conda install --yes --quiet -c conda-forge -c anaconda \
    python=$PYTHON_VERSION \
    xrootd \
    tensorflow-gpu \
    && conda clean --yes --all --force-pkgs-dirs 

RUN conda run pip install --quiet \
  snakemake \
  pandas \
  uproot \
  numpy \
  tensorflow-gpu==2.3.1 \
  scikit-learn 

RUN conda run pip install --quiet scikinC scipy==1.4.1

ENV USER='Dockerman'

CMD git clone -b fastfortests git@github.com:landerlini/lb-trksim-train.git trk \
    && git clone -b test git@github.com:mbarbetti/lb-pidgan-train.git pid \
    && echo "### INSTALLING ###" \ 
    && cd /trk && python setup.py bdist_wheel && pip install dist/*.whl \
    && echo "### LAUNCHING ###" \ 
    && snakemake -s Snakefile -j 32 compile_2016-MagUp   \
    && cd /pid/lb-pidgan-train && snakemake -j32 all \
    && echo "### DONE ###"  
     
