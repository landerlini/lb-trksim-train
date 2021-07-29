# LHCb Tracking Parametrization Training
Scripts and logics to train tracking models for the ultra-fast simulation of the LHCb experiment.

The ultra-fast simulation of the LHCb experiment is based on a combination 
of simple parametrization and machine-learning models that concur to the 
parametrization of the overall response of the whole detector. 

We organize in this package the code to train and validate the machine-learning 
models used to parametrize the response of the tracking system.

This includes:
 * the geometrical acceptance of the detector
 * the tracking efficiency 
 * the resolution on the position and the momentum of the reconstructed tracks
 * the covariance matrix of the track parameters in its closest approach to the 
   beam direction

### See also
 * `lb-pidgan-train` for the training of the models describing the PID  

### Dependencies
 * NumPy 
 * Pandas 
 * Uproot [[scikit-hep/uproot4](https://github.com/scikit-hep/uproot4)] 
 * TensorFlow 2 
 * Scikit-learn
 * HTML Reports [[html-reports](https://github.com/villoro/html-reports)]
 * scikinC [[scikinC](https://github.com/landerlini/scikinC)]


### How to train the models
The pipeline is described in a Snakefile so 
```
snakemake -j <number-of-cores> all 
```
should be enough to train the whole set of models.




