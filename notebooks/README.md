# Modeling the LHCb tracking with Deep Neural Networks

This package defines the training procedure for the neural networks contributing to the
parametrization of the LHCb tracking performance as a pipeline of notebooks. 

The pipeline is defined as a Direct Asynchronous Graph (DAG) using Snakemake, 
according to the rules presented in the [Snakefile](./Snakefile).

The logical flow of the tracking parametrization is as follows:

 * Evaluate whether a particle is in geometrical acceptance
 * Evaluate whether a particle is reconstructed as a track, and of which 
   `track-type` (long, upstream or downstream)
 * Smear the Monte Carlo parameters into track parameters according to a 
   resolution function that may depend on the kinematics of the particle
 * Assess the reconstruction uncertainty on the track, encoded into the covariance 
   matrix of the Closest-To-Beam track state, as a function of both kinematic 
   features and measures of the track reconstruction quality (*e.g.* the track &chi;<sup>2</sup>).

The notebooks are divided in two main categories: 

 * **Filter functions**, defining the probability of a particle to survive some fixed selection, such as 
     being in geometrical acceptance or pass the track quality requirements 
 * **Feature functions**, defining new features with arbitary distribution and therefore involving the 
     usage of generative models (and in particular GANs).
     
## Filter functions (acceptance and tracking efficiency)
 * Data preprocessing: [Preprocessing.ipynb](./Preprocessing.ipynb)
 * Training of the acceptance model: [Acceptance.ipynb](./Acceptance.ipynb)
 * Validation of the acceptance model: [Acceptance-validation.ipynb](./Acceptance-validation.ipynb)
 * Training of the efficiency model: [Efficiency.ipynb](./Efficiency.ipynb)
 * Validation of the efficiency model: [Efficiency-validation.ipynb](./Efficiency-validation.ipynb)
 
## Feature functions
 * Data preprocessing: [Preprocessing-GANs.ipynb](./Preprocessing-GANs.ipynb)
 * Training of the resolution model: [Resolution.ipynb](./Resolution.ipynb)
 * Validation of the resolution model [Resolution-validation.ipynb](./Resolution-validation.ipynb)
 * Training of the covariance model: [Covariance.ipynb](./Covariance.ipynb)
 * Validation of the covariance model: [Covariance-validation.ipynb](./Covariance-validation.ipynb)

## Deploying the trained models
The deployment of the trained model relies on [scikinC](https://github.com/landerlini/scikinC)
as described in the [Deploy.ipynb](./Deploy.ipynb) notebook.

# Running the code
To run the code you will need to set up the environment. 
It is recommended to use the Docker image [landerlini/lhcbaf:v0p8](https://hub.docker.com/r/landerlini/lhcbaf)
defining two conda environment:

 * **LHCb Analysis Facility**, to be used for preprocessing and validation notebooks
 * **TensorFlow on GPU**, to be used for training
 
Using the docker image, the whole pipeline runs issueing

```bash
snakemake --resources gpu=1
```

from the folder containing this `README.md` file.


