"""
Functions useful to validate the models, mimicking in python some steps happening in Lamarr
"""
import numpy as np

def invert_column_transformer(column_transformer, preprocessed_X):
    """
    Invert a column transformer (works only for N -> N transformations)
    """
    preprocessed_split = {name: [None]*len(cols) for name, _, cols in column_transformer.transformers_}
        
    for iCol in range(preprocessed_X.shape[1]):
        name, transformer, cols = [(n, t, list(c)) for n, t, c in column_transformer.transformers_ if iCol in c].pop()
        preprocessed_split[name][cols.index(iCol)] = preprocessed_X[:,iCol]
    
    X = []
    for name, algo, _ in column_transformer.transformers_:
        split = np.stack(preprocessed_split[name], axis=1)
        X.append(split if algo == 'passthrough' else algo.inverse_transform(split))
    
    return np.concatenate(X, axis=1)

