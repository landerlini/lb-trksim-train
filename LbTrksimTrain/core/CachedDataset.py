from .Dataset import Dataset 
from logging import getLogger as logger

class CachedDataset (Dataset): 
  def __init__ (self, config, entrysteps = 10000000, max_chunks = 100000, max_files = 1000):
    self.config = config 
    self.entrysteps = entrysteps
    self.max_files = max_files 
    self.max_chunks = max_chunks 
    self.cached_chunks = None 

  def iterate (self, config = None):
    config = config or self.config 
    if self.cached_chunks is None:
      logger('CachedDataset').info ( "Caching dataset. Please wait") 
      self.cached_chunks = {iChunk: chunk for iChunk, chunk in enumerate(
          Dataset.iterate(self.config, self.entrysteps, self.max_chunks, self.max_files, forever=False)
          )}

    for iChunk in range (len(self.cached_chunks)):
      yield self.cached_chunks[iChunk] 

  def clear_cache(self):
    logger('CachedDataset').info ( "Cache cleared") 
    self.cached_chunks = None





