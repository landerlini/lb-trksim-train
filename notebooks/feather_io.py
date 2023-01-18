"""
feather_io.py - Utility functions for accessing feather datasets split in multiple files
Author: Lucio.Anderlini@fi.infn.it

All files of the same dataset are supposed to be stored in a unique, dedicated directory
with a `definition.json` file defining names of features and labels.

#### Example

```python
import numpy as np
import pandas as pd
import dask.dataframe as ddf

## Consider some dataset including labels and features stored in a dask dataframe
columns = ['x1', 'x2', 'x3', 'y']
df = pd.DataFrame(np.random.normal(0, 1, (1_000_000, len(columns))).astype(np.float32), columns=columns)
random_data = ddf.from_pandas(df, npartitions=10)

## Write the dataframe to a folder
from feather_io import FeatherWriter
MiB = int(2**20)
fw = FeatherWriter("random_data/", features=columns[:-1], labels=columns[-1:])
partition_lengths = (
    random_data
    .repartition(partition_size=5*MiB) ## define here the size of each file
    .map_partitions(fw) 
    .compute() 
)

## Reload the dataset to dask
from feather_io import FeatherReader
reader = FeatherReader("random_data/")
loaded = reader.as_dask_dataframe()


loaded = reader.as_tf_dataset()
## Reload as tensorflow dataset


```

"""

import numpy as np
import pandas as pd
import os, json
from glob import glob
import tensorflow as tf
import dask.dataframe as ddf
import shutil

class FeatherWriter:
    """
    Store a dask dataframe as a set of feather files
    """
    def __init__ (self, output_dir: str, features: tuple, labels: tuple, preprocessorX=None, preprocessorY=None):
        self.output_dir = output_dir
        self.features = list(features)
        self.labels = list(labels)
        self.preprocessorX = preprocessorX
        self.preprocessorY = preprocessorY

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        with open(os.path.join(output_dir, "definitions.json"), "w") as f:
            json.dump(dict(features=features, labels=labels), f)
                  
    
    def __call__(self, df: pd.DataFrame):
        X = df[self.features].values
        y = df[self.labels].values
        prepX = self.preprocessorX.transform(X) if self.preprocessorX is not None else X
        prepY = self.preprocessorY.transform(y) if self.preprocessorY is not None else y
        
        partition = pd.DataFrame(np.c_[prepX, prepY], columns=self.features+self.labels)
        partition.to_feather(os.path.join(self.output_dir, f"{np.random.randint(0, 0xFFFFFFFF):08x}.feather"))
                                                
        return len(partition)


class FeatherReader:
    """
   Loads feather datasets stored in multiple files as dask dataframes or tf.data.Datasets
    """
    def __init__ (self, input_dir: str):
        with open(os.path.join(input_dir, "definitions.json")) as f:
            definitions = json.load(f)
    
        self._features = definitions['features']
        self._labels = definitions['labels']
        self._data_files = glob(os.path.join(input_dir, "*.feather"))
        
    @property
    def features(self):
        return self._features
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def data_files(self):
        return self._data_files
    
    @property
    def output_signature(self):
        return (
            tf.TensorSpec(shape=(None, len(self.features)), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(self.labels)), dtype=tf.float32))

    def tf_generator(self):
        for filename in self.data_files:
            with open(filename, 'rb') as f:
                df = pd.read_feather(filename)

            yield (tf.constant(df[self.features].values), tf.constant(df[self.labels].values))
    
    def dask_loader(self, filename):
        return pd.read_feather(open(filename, 'rb'))
    
    def as_tf_dataset(self):
        return tf.data.Dataset.from_generator(
            self.tf_generator, 
            output_signature=self.output_signature
        ).unbatch()
    
    def as_dask_dataframe(self):
        return ddf.from_map(self.dask_loader, self.data_files)
    


