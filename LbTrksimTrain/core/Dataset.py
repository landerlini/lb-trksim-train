import uproot 
from glob import glob
import logging 
import numpy as np 
import pandas as pd
import re 
import math 
from tqdm import tqdm 

class Dataset:
  def __init__ (self, config):
    mydb = next(iter( self.iterate(config) ))
    self.config = config 

    print (mydb.describe()) 


  @staticmethod 
  def iterate (config, entrysteps = 1000000, max_chunks = 10000, max_files = 1000, forever=True):
    while True:
        branches = Dataset.list_branches ( config )
        filelist = Dataset.get_filelist (config['files'])[:max_files]
        for iChunk, chunk in enumerate( uproot.iterate (
              {f: config['treename'] for f in filelist}, 
              entrysteps = entrysteps,
              expressions = branches, 
              library='pd',
            )): 
          logging.getLogger("Dataset").info("Processing chunk %d with %d entries" % (iChunk, len(chunk))) 
          if 'baseline_cut' in config.keys(): 
            chunk.query ( " and ".join ( config('baseline_cut', []) ), inplace = True ) 
            logging.getLogger("Dataset").info("Chunk length after baseline requirements: %d entries" % (len(chunk))) 
          chunk.drop (columns = [b for b in branches if b not in config.variables.values()], inplace = True)
          Dataset.rename_columns ( config, chunk ) 

          if iChunk < max_chunks:
            yield chunk 
          else:
            return None 
          
        if not forever: 
          break

  def get (config, selection = None, max_chunks=10000, entrysteps=10000000, max_files = 1000): 
    branches = Dataset.list_branches ( config )
    all_dbs = [] 
    filelist = Dataset.get_filelist (config['files'])[:max_files] 
    for iChunk, chunk in enumerate( uproot.iterate ( 
          {f: config['treename'] for f in filelist}, 
          entrysteps = entrysteps, #config('entrysteps', entrysteps), 
          expressions = branches,  
          library='pd',
        )): 
      logging.getLogger("Dataset").info("Processing chunk %d" % iChunk) 
      if 'baseline_cut' in config.keys(): 
        chunk.query ( " and ".join ( config('baseline_cut', []) ), inplace = True ) 
      if selection is not None: 
        chunk.query ( selection, inplace = True ) 
      chunk.drop (columns = [b for b in branches if b not in config.variables.values()], inplace = True)
      Dataset.rename_columns ( config, chunk ) 

      all_dbs.append ( chunk ) 
    return pd.concat (all_dbs)


  @staticmethod
  def list_branches ( config ):
    requested_vars = config['variables'].values() 
    cut_vars = [] 
    for key in config.keys():
      if "cut" in key or "requirement" in key:
        cuts = [config[key]] if isinstance (config[key], str) else config[key] 
        for cut in cuts:  
          cut_vars += [v for v in re.findall ( r"[A-Za-z_]*", cut ) if v != ''] 


    ret = list(requested_vars) + [v for v in cut_vars if not hasattr (math, v)] 
    return list(set(ret))



  @staticmethod 
  def rename_columns ( config, db ):
    original_columns = db.columns 
    print ("originally", original_columns)
#    print ("#"*80)
#    print ("\n".join(original_columns)) 
#    print ("#"*80)
    new_names = {v:k for k,v in config.variables.items()} 

    for column in db.columns:
      if column.count('[') == 3:
        name, i = (re.findall ("([A-Za-z_]*)\[([0-9]*)\]", column)[0] ) 
        new_names [column] = "%s_%d" % (new_names[name],i) 
      elif column.count('[') == 2:
        name, i, j = (re.findall ("([A-Za-z_]*)\[([0-9]*)\]\[([0-9]*)\]", column)[0] ) 
        new_names [column] = "%s_%s_%s" % (new_names[name],i,j) 

    db.columns = [new_names[old_name] if old_name in new_names.keys() else old_name for old_name in original_columns] 
    print ("and then", db.columns)
#    print (">"*80)
#    print ("\n".join(db.columns)) 
#    print ("<"*80)





  def hist ( self, formula, selection = None, binning = None, **kwargs): 
    hists = [] 
    if isinstance (selection, (list,tuple)): 
      selection = " and ".join ( ["(%s)" % cut for cut in selection] ) 

    edges = None
    if isinstance(binning, (list,tuple)): 
      edges = np.linspace ( binning[1], binning[2], binning[0]+1 ) 

    for iChunk, chunk in enumerate(tqdm(self.iterate (self.config, **kwargs))):
      if selection is not None: 
        selected = chunk.query (selection)
        if len(selection) == 0: 
          print ("WARNING! Empty chunk")
          hists.append ( np.zeros ( len(edges)-1, dtype = np.float64 ) ) 
          continue 

      if edges is None and isinstance(binning, (type(None), int)):
        mydb = next(iter(self.iterate(self.config)))
        if binning is None: nbins = 100 
        var = mydb.eval ( formula ) 
        edges = np.linspace ( var.min(), var.max(), int(nbins) ) 

      var = selected.eval ( formula ) 
      print (formula, var.min(), var.max()) 
      #print (np.quantile ( var, np.linspace (0,1,11) ))
      hists.append ( np.histogram ( var,
            bins = edges
          ) [0]
        )

    return edges, np.sum ( np.array (hists), axis = 0 )
    
  
  @staticmethod
  def get_filelist ( list_of_strings ):
    return sum([glob(s) for s in list_of_strings], []) 

  def count ( self, selection = None ):
    cfg = self.config 
    count = uproot.numentries ( 
        {f: self.config['treename'] for f in self.config['files']}, 
        )

    if selection is None:
      return count

    cut_vars = [v for v in re.findall ( r"[A-Za-z_]*", selection ) if v != '' and not hasattr(math,v)] 

    selcount = 0 
    for iChunk, chunk in enumerate( uproot.iterate ( 
          {f: self.config['treename'] for f in self.config['files']}, 
          entrysteps = 1000000,
          expressions = cut_vars,  
          library='pd',
        )): 
      selcount += len ( chunk.query ( selection ) )

    return selcount 










