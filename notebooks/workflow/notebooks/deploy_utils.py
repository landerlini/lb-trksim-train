"""
Important disclaimer.
This file is a collection of ugly hacks. 
Please do not copy-paste them elsewhere, or if you do, don't blame me.
"""

import os, os.path
import pickle
import numpy as np
import tensorflow as tf
import sklearn
import scikinC, scikinC.layers


class hacks:
    
    @staticmethod
    def copy_dense(layer_from, layer_to):
            layer_to.build(input_shape=layer_from.input.shape)
            layer_to.kernel.assign(layer_from.kernel)
            layer_to.bias.assign(layer_from.bias)
            layer_to.activation = layer_from.activation

    class DenseWithSkipConnection (tf.keras.layers.Dense):
        """
        A hacky version of a dense layer with a skip connection in a single layer
        """
        def __init__(self, dense_layer):
            tf.keras.layers.Dense.__init__(
                self, 
                dense_layer.bias.shape[0],
                input_shape=dense_layer.input.shape,
                name=f"res{dense_layer.name.lower()}"
            )
            
            hacks.copy_dense(layer_from=dense_layer, layer_to=self)            
        
        def call (self, inputs):
            return inputs + tf.keras.layers.Dense.call(self, inputs)

    class scikinC_DenseWithSkipConnection (scikinC.layers.Dense):
        """
        A *very* hacky scikinC description of the dense layer with a skip connection.
        Note that, it relies on the name ("input") given to the C implementation of the 
        dense transform in scikinC.layers.Dense. 
        
        Changing that name would unexpectedly kill this hack.
        """
        def activate (self, x):
            std_activate = scikinC.layers.Dense.activate(self, x)
            std_activate += "\n %(x)s = %(x)s + %(inp)s;" % {'x':x, 'inp': x.replace('ret', 'input')}
            return std_activate
        

def count_model_weights (model):
    "Counts the number of weights in a dense network"
    return sum([
        sum([np.prod(w.shape) for w in l.trainable_weights]) 
        for l in model.layers])
        
class LamarrModel:
    """
    Collection of a preprocessing step, a model and a postprocessing step.
    
    Tries to collapse models with dense layers with a skip connection 
    to make it possible to convert them to C with scikinC.
    """
    def __init__(self, model, tX, tY=None):
        self._model = model
        self._tX = tX
        self._tY = tY
        self._collapsed_model = self.collapse_model(model)
        
    @property
    def model(self):
        return self._model
    
    @property
    def tX(self):
        return self._tX
    
    @property
    def tY(self):
        return self._tY
    
    @property
    def collapsed_model(self):
        return self._collapsed_model
    
    @property
    def pipeline(self):
        algos = []
        if self.tX is not None:
            algos.append(("preprocessing", self.tX))
        algos.append(("dnn", self.collapsed_model))
        
        if self.tY is not None:
            algos.append(("postprocessing", self.tX))
            
        return sklearn.pipeline.Pipeline(algos)
    
    @staticmethod
    def from_saved_model_pb(filename):
        model_dir = os.path.dirname(filename)
        tX = tY = None
        if os.path.exists(os.path.join(model_dir, "tX.pkl")):
            tX = pickle.load(open(os.path.join(model_dir, "tX.pkl"), 'rb'))
        if os.path.exists(os.path.join(model_dir, "tY.pkl")):
            tY = pickle.load(open(os.path.join(model_dir, "tY.pkl"), 'rb'))
    
        print (
            f"Loading model from '{model_dir}'.  "
            f"Preprocessing: {'👌' if tX is not None else '😞'}." 
            f"Postprocessing: {'👌' if tY is not None else '😞'}." 
        )
        return LamarrModel(tf.keras.models.load_model(filename), tX=tX, tY=tY)
            
    @staticmethod
    def collapse_model (model):
        collapsed_layers = []
        layer_seq = model.layers + [None]
        for layer, next_layer in zip(layer_seq[:-1], layer_seq[1:]):
            if 'input' in layer.name.lower(): continue 
            if 'concatenate' in layer.name.lower(): continue
            if 'add' in layer.name.lower(): continue
            # identify a skip connection:
            if (next_layer is not None and
                'add' in next_layer.name.lower() and 
                'dense' in layer.name.lower()
               ):
                collapsed_layers.append(hacks.DenseWithSkipConnection(layer))
            elif 'dense' in layer.name.lower():
                collapsed_layers.append(tf.keras.layers.Dense( 
                    layer.bias.shape[0],
                    input_shape=layer.input.shape[1:],
                    name=f"cpy_{layer.name.lower()}"
                ))
                hacks.copy_dense(layer_from=layer, layer_to=collapsed_layers[-1])
            else:
                collapsed_layers.append(layer)
                

        collapsed_model = tf.keras.models.Sequential(collapsed_layers)
        LamarrModel.raise_on_unequal_number_of_weights(model, collapsed_model)

        if len(model.inputs) == 1:
            nNodes = model.input_shape[1]
        else:
            nNodes = sum([inp_shape[1] for inp_shape in model.input_shape])

        collapsed_model.predict([
            np.random.normal(0, 1, [1, nNodes]),
        ], verbose=False)

        return collapsed_model
    
    @staticmethod
    def raise_on_unequal_number_of_weights(model, collapsed_model):
        n_weights_original = count_model_weights(model)
        n_weights_collapsed = count_model_weights(collapsed_model)

        msg = (
            f"Original model: {n_weights_original}. "
            f"Collapsed model: {n_weights_collapsed}. "
        )

        if n_weights_original == n_weights_collapsed:
            print (f"Check on the number of weights: ✅! {msg}")
            
        else:
            raise ValueError(f"Model conversion failed. {msg}")
            

    

   
