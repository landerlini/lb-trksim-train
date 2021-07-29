import matplotlib.pyplot as plt 
from LbTrksimTrain.core import CachedDataset 

def plot (cfg, report): 
  datasets = {name: CachedDataset(cfg.datasets[name],max_chunks = 10, entrysteps = 10000,) for name in [ "Lamarr", "BrunelGenerated" ]}
  report . add_markdown ("""
      As a first step we ensure that the distributions at generator level are
      consistent for the two samples. 
      """)
  for category, selection in cfg.categories.items(): 
    report.add_markdown ( "### %s " % category ) 
    for variable in ['x', 'y', 'z', 'tx', 'ty', 'logpz' ]:
      varcfg = cfg.variables[variable] 
      for datasetName, dataset in datasets.items(): 
        edges, contents = dataset.hist ( varcfg('formula', variable),  binning = varcfg.binning, selection = selection ) 
        if 'unit' in varcfg.keys(): 
          plt.xlabel ( "%s [%s]" % (varcfg.title, varcfg.unit) )
          plt.ylabel ( "Entries / [ %.2f %s ]" % (edges[1]-edges[0], varcfg.unit))
        else: 
          plt.xlabel ( "%s" % (varcfg.title) )
          plt.ylabel ( "Entries / %.2f" % (edges[1]-edges[0]) )

        print (edges.shape, contents.shape) 
        plt.hist ( 0.5*(edges[1:]+edges[:-1]), bins = edges, weights = contents, density = True, 
            linewidth = 2, 
            histtype = 'bar' if 'Lamarr' == datasetName else 'step', label = cfg.datasets[datasetName].title ) 

      plt.legend() 
      plt.yscale ( 'log' ) 
      plt.title ( category ) 
      report.add_figure(options = 'width = 32%') ; plt.clf() 

