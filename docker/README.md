# Docker

Docker deployment of the training pipeline.
Currently we lack a way for accessing data remotely, so it will only work on the INFN Florence cluster, by running:
```
sudo docker run -it -v /home/lucio/.ssh:/root/.ssh -v /cvmfs:/cvmfs -v /pclhcb06/landerli:/pclhcb06/landerli/ landerlini/lamarr-train:v0.1.26
```
