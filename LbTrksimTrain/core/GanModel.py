import os 
import pickle 

import pickle 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer , MinMaxScaler
from functools import partial


class DecorrTransformer ():
  def fit ( self, X, y = None ):
    self.cov = np.cov (X.T)
    _, self.eig = np.linalg.eig ( self.cov )
    return self

  def transform (self, X):
    dX = X.dot (self.eig) 
    return dX 

  def inverse_transform (self, dX): 
    X = dX.dot (self.eig.T) 
    return X 


class GanModel: 
  def __init__ (self, 
                n_iterations=100, 
                preprocessing=None, 
                n_generator_layers=3, 
                n_generator_nodes=128, 
                n_discriminator_layers=3, 
                n_discriminator_nodes=128, 
                n_noise_inputs=64,
                generator_optimizer=None,
                discriminator_optimizer=None,
                bootstrap=True,
                batchsize=1000., 
                lambda_=0.3, 
                gamma_=1.0, 
                discriminator_learning_rate  = 1e-5,
                generator_learning_rate = 1e-5,
                wreferee = 5, 
                generator_lazyness = None,
                ):

    self.preprocessing = preprocessing or [
      ('minmax', MinMaxScaler ),
      ('quantile1', partial (QuantileTransformer, output_distribution = 'normal') ),
      ('decorrelate', DecorrTransformer ),
      ('quantile2', partial (QuantileTransformer, output_distribution = 'normal') ),
      ]

    ## 
    self.n_iterations_                = n_iterations
    self.n_generator_layers_          = n_generator_layers
    self.n_generator_nodes_           = n_generator_nodes
    self.n_discriminator_layers_      = n_discriminator_layers
    self.n_discriminator_nodes_       = n_discriminator_nodes
    self.n_noise_inputs_              = n_noise_inputs
    self.generator_optimizer_         = generator_optimizer or Adam (generator_learning_rate, beta_1=0.5, beta_2=0.9) 
    self.discriminator_optimizer_     = discriminator_optimizer or Adam (discriminator_learning_rate, beta_1=0.5, beta_2=0.9) 
    self.generator_learning_rate_     = generator_learning_rate
    self.discriminator_learning_rate_ = discriminator_learning_rate
    self.wreferee_                    = wreferee
    self.bootstrap_                   = bootstrap 
    self.batchsize_                   = batchsize 
    self.lambda_                      = lambda_ 
    self.gamma_                       = gamma_ 
    self.generator_lazyness_          = generator_lazyness

    self.generator_ = tf.keras.models.Sequential() 
    for iLayer in range(self.n_generator_layers_): 
      self.generator_.add(
          tf.keras.layers.Dense ( self.n_generator_nodes_, 
            kernel_initializer='he_normal', 
            activation='tanh') 
          )

    self.discriminator_ = tf.keras.models.Sequential() 
    for iLayer in range(self.n_discriminator_layers_):
      self.discriminator_.add( 
          tf.keras.layers.Dense(self.n_discriminator_nodes_, 
            kernel_initializer='he_normal', 
            activation='tanh') 
          )
#      self.discriminator_.add(
#            tf.keras.layers.Dropout(0.5)
#          )

    self.discriminator_.add(
        tf.keras.layers.Dense (1, 
          kernel_initializer='he_normal', 
          activation='linear') 
        )


  def fit( self, X, Y, w = None ): 
    if Y.shape[0] != X.shape[0]: 
      raise ValueError ("Inconsistent shape of X (%(shpX)s) and Y (%(shpYs)" % 
          {'shpX': str(X.shape), 'shpY': str(Y.shape)} ) 

    if Y.shape[1] > self.n_noise_inputs_:
      raise ValueError ("Insufficient noise inputs (%d) to represent"
                        "a target tensor of shape %s" 
                        % (self.n_noise_inputs_, str(Y.shape) ))


    self.batch_size_ = X.shape[0]
    self.n_X_ = X.shape[1]
    self.n_Y_ = Y.shape[1]

    if len(self.generator_.layers) == self.n_generator_layers_:
      self.generator_.add( 
          tf.keras.layers.Dense (self.n_Y_, 
            kernel_initializer='he_normal', 
            activation='linear')
          )

    X, Y = self._apply_preprocessing(X, Y)

#    if not hasattr (self, '_training_step_'): 
#      self._training_step_ = tf.function ( self._training_step,
#        input_signature = [
#              tf.TensorSpec(shape=[None, self.n_X_], dtype=tf.float32), 
#              tf.TensorSpec(shape=[None, self.n_Y_], dtype=tf.float32), 
#              tf.TensorSpec(shape=[None], dtype=tf.float32), 
#            ]) 
      
    self.losses_ = [] 
    for i_iteration in range(self.n_iterations_):
      X_, Y_, w_ = self._sample(X, Y, w) 
      self.losses_.append(self._training_step ( 
        X_.astype ( np.float32 ), 
        Y_.astype ( np.float32 ),
        w_.astype ( np.float32 )
        ))

    return self.losses_

  
  def _sample (self, X, Y, w):
    if self.bootstrap_ is False and self.batchsize_ >= len(X):
      if w is None: 
        w = np.ones ( len(X) )
      return X, Y, w

    idx = np.random.randint ( 0, len(X), self.batchsize_ )
    return (X[idx], Y[idx], w[idx]) if w is not None else (X[idx], Y[idx], np.ones(len(idx)))


  def _apply_preprocessing (self, X, Y):
    if not hasattr(self, 'transformerX_') or not hasattr(self, 'transformerY_'):
      self.transformerX_ = Pipeline ( 
          [(name, step()) for name,step in self.preprocessing] 
          )
      self.transformerX_.fit ( X )

      self.transformerY_ = Pipeline ( 
          [(name, step()) for name,step in self.preprocessing] 
          )
      self.transformerY_.fit ( Y )

    tX = self.transformerX_.transform (X) 
    tY = self.transformerY_.transform (Y) 
    print (f"PREPROCESSING:  X {X.dtype} -> tX {tX.dtype}")
    print (f"                Y {Y.dtype} -> tY {tY.dtype}")
    return tX, tY 


  @tf.function 
  def _training_step (self, X, Y, w, train_generator=True): 
    xentropy  = tf.keras.losses.BinaryCrossentropy (label_smoothing = 0.1, from_logits = True)
    ##
    noise = tf.random.normal ((tf.shape(X)[0], self.n_noise_inputs_))
    rX = tf.concat ( [X, noise], axis = 1 )
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      Yhat = self.lambda_ * self.generator_(rX, training = True) + self.gamma_ * rX[:,-self.n_Y_:] 
      ##
      real_d    = self.discriminator_(tf.concat([X, Y], axis=1), training=True)
      real_loss = xentropy (tf.ones_like(real_d), real_d, sample_weight = w) 
      ## 
      fake_d    = self.discriminator_(tf.concat([X, Yhat], axis=1), training=True)
      fake_loss = xentropy (tf.zeros_like(fake_d), fake_d, sample_weight = w) 

      dLoss = 0.5*(real_loss + fake_loss)
      gLoss = xentropy ( tf.ones_like (fake_d), fake_d, sample_weight = w ) #-tf.reduce_mean ( fake_d )

    self.discriminator_.learning_rate = self.discriminator_learning_rate_ #* tf.nn.sigmoid ( (dLoss - np.log(2))*self.wreferee_ )
    self.generator_.learning_rate = self.generator_learning_rate_ #* tf.nn.sigmoid ( -(dLoss - np.log(2))*self.wreferee_ )

    dGrads = disc_tape.gradient(dLoss, self.discriminator_.trainable_variables) 
    gGrads = gen_tape.gradient(gLoss, self.generator_.trainable_variables) 

    self.discriminator_optimizer_.apply_gradients(
        zip(dGrads, self.discriminator_.trainable_variables)) 

    self.generator_optimizer_.apply_gradients(
        zip(gGrads, self.generator_.trainable_variables)) 

    return dLoss 


  def predict (self, X):
    if not hasattr (self, 'transformerX_') or not hasattr(self,'transformerY_'):
      raise RuntimeError ("GanModel was never trained or has been reset")

    tX = self.transformerX_.transform ( X )
    noise = np.random.normal(0, 1, (len(X), self.n_noise_inputs_))
    rX = np.concatenate ([tX,noise], axis=1)
    Yhat = self.lambda_ * self.generator_(rX).numpy() + rX[:,-self.n_Y_:] 

    return self.transformerY_.inverse_transform(Yhat) 


  def save (self, export_path):
    self.generator_.save ( export_path )
    for key in ['transformerX_', 'transformerY_', 'lambda_', 'n_noise_inputs_', 'n_X_', 'n_Y_']:
      with open ( os.path.join(export_path, key), 'wb') as f: 
        pickle.dump ( getattr(self, key), f ) 

  @staticmethod 
  def load (export_path):
    ret = GanModel() 
    ret.generator_ = tf.keras.models.load_model (export_path, compile=False) 
    for key in ['transformerX_', 'transformerY_', 'lambda_', 'n_noise_inputs_', 'n_X_', 'n_Y_']:
      with open ( os.path.join(export_path, key), 'rb') as f: 
        setattr(ret, key, pickle.load ( f )) 

    return ret 

    

