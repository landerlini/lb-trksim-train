"""
Makes preliminary plots to visualize the geometrical acceptance
"""
import numpy as np 
import matplotlib.pyplot as plt 
from LbTrksimTrain.core import CachedDataset 
from LbTrksimTrain.core.Report import Jscript  
import pickle
from tqdm import trange
from logging import getLogger as logger

from LbTrksimTrain.core import make_histogram 


def plot (cfg, report): 
  generated = CachedDataset(cfg.datasets['BrunelGenerated'],  max_chunks = 10, entrysteps = 100000, files_key='files_train') 

  for state, stcfg in cfg.states.items():
    nh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 
    dh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 
    binning = [np.linspace(stcfg.binning[1], stcfg.binning[2], stcfg.binning[0]+1)]*2 
    for db in generated.iterate():
      dh += np.histogram2d ( db['x_%s'%state ], db['y_%s'%state], bins = binning)[0]
    for db in generated.iterate():
      sdb = db.query ( "acceptance == 1")
      nh += np.histogram2d ( sdb['x_%s'%state ], sdb['y_%s'%state], bins = binning)[0]

    ratio = np.where (dh > 0, nh/dh, np.min(nh/(dh+1)))
    plt.imshow (ratio.T, extent = [binning[0][0],binning[0][-1],binning[1][0],binning[0][-1]], origin = 'lower') 
    plt.title ( "%s" % state ) 
    plt.xlabel ( "x coordinate [mm]" ) 
    plt.ylabel ( "y coordinate [mm]" ) 
    report.add_figure(options = 'width = 49%') ; plt.clf() ; plt.close() 





def train (cfg, report, modelfile): 
  from sklearn.ensemble import GradientBoostingClassifier
  from sklearn.utils import shuffle 
  from itertools import cycle

  report.add_markdown ( "### Training acceptance BDT")
  generated = CachedDataset(cfg.datasets['BrunelGenerated'], max_chunks = 10, entrysteps = 10000000, files_key='files_train') 

  accBdt = cfg.acceptanceBDT 

  classifier = GradientBoostingClassifier(n_estimators = 1, warm_start = True, 
        subsample = cfg.acceptanceBDT.subsample, 
        learning_rate = cfg.acceptanceBDT.learning_rate,
        max_depth = cfg.acceptanceBDT.max_depth, 
      )

  testscore = []
  trainscore = [] 
  dbs = iter(cycle(generated.iterate())) 
  othVars = ['acceptance', 'pz'] 
  for i in trange (cfg.acceptanceBDT.nEpochs, ncols = 80, ascii = True):
    db = next(dbs) 
    db.drop(columns = [c for c in db.columns if c not in accBdt.discrVars + othVars], inplace=True)
    db = shuffle ( db )
    db.reset_index ( drop=True, inplace = True )

    training, test = db[:len(db)//2], db[len(db)//2:] 
    classifier.n_estimators += 1
    classifier.fit ( training[accBdt.discrVars].values, training['acceptance'].values,  )
    #trainscore.append( classifier.train_score_[-1] )
    strain = training.sample(1000) if len(training) > 1000 else training
    stest  = test.sample(1000) if len(test) > 1000 else test
    trainscore .append( classifier.score ( strain[accBdt.discrVars], strain['acceptance']) )
    testscore  .append( classifier.score ( stest[accBdt.discrVars], stest['acceptance']) )

  with open  (modelfile, 'wb') as fout:
    pickle.dump ( classifier, fout )

  plt.plot ( trainscore, label = "Train" ) 
  plt.plot ( testscore, label="Validation" ) 
  plt.xlabel ( "Epochs" ) 
  plt.ylabel ( "Score" ) 
  plt.legend() 
  report.add_figure(); plt.clf() 


  bins = np.linspace ( -2, 8, 101 ) 
  pdb = next(dbs)
  var = np.log(np.maximum(pdb['pz'].values,1e-3))/np.log(10)
  acc = pdb ['acceptance'] 
  plt.hist ( var[acc], bins = bins, label = 'Selected', density = True )
  plt.hist ( var, bins = bins, label = 'Unweighted', density = True, histtype = 'step', linewidth = 2, linestyle = '--' )
  plt.hist ( var, bins = bins, weights = classifier.predict_proba(pdb[accBdt.discrVars].values)[:,1], label = 'Weighted', histtype = 'step', linewidth = 2, color = 'red', density = True )
  plt.title ( "Acceptance" )
  plt.xlabel ( "$\log_{10}$ (Longitudinal momentum / 1 MeV)" )
  plt.ylabel ( "Normalized candidates" )
  plt.legend() 
  report.add_figure(); plt.clf() 

    
    

def validate (cfg, report, modelfile): 
  accBdt = cfg.acceptanceBDT 

  ## Load the trained model 
  with open  (modelfile, 'rb') as fin:
    classifier = pickle.load ( fin )


  #generated = CachedDataset(cfg.datasets['BrunelGenerated'],  max_chunks = 100, entrysteps = 100000,) 
  generated = CachedDataset(cfg.datasets['BrunelGenerated'],  max_chunks = 100, entrysteps = 100000, files_key='files_validate') 

  for state, stcfg in cfg.states.items():
    ########################## SIMULATED xy ###########################################
    nh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 
    dh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 
    binning = [np.linspace(stcfg.binning[1], stcfg.binning[2], stcfg.binning[0]+1)]*2 
    for db in generated.iterate():
      dh += np.histogram2d ( db['x_%s'%state ], db['y_%s'%state], bins = binning)[0]
    for db in generated.iterate():
      sdb = db.query ( "acceptance == 1")
      nh += np.histogram2d ( sdb['x_%s'%state ], sdb['y_%s'%state], bins = binning)[0]

    ratio = np.where (dh > 0, nh/dh, np.min(nh/(dh+1)))
    plt.imshow (ratio.T, extent = [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]], origin = 'lower') 
    plt.title ( "%s (Boole/Brunel reconstruction)" % state ) 
    plt.xlabel ( "x coordinate [mm]" ) 
    plt.ylabel ( "y coordinate [mm]" ) 
    js = Jscript().hist2d(f"{state}_xy_brunel", ratio.T, [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]])
    report.add_figure(options=str(js.width("49%"))) ; plt.clf() ; plt.close() 


    ########################## GENERATED xy ###########################################
    nh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 
    dh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 

    binning = [np.linspace(stcfg.binning[1], stcfg.binning[2], stcfg.binning[0]+1)]*2 
    for db in generated.iterate():
      dh += np.histogram2d ( db['x_%s'%state ], db['y_%s'%state], bins = binning)[0]
    for db in generated.iterate():
      w = classifier.predict_proba( db[accBdt.discrVars].values )[:,1]
      nh += np.histogram2d ( db['x_%s'%state ], db['y_%s'%state], bins = binning, weights=w)[0]

    ratio = np.where (dh > 0, nh/dh, np.min(nh/(dh+1)))
    plt.imshow (ratio.T, extent = [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]], origin = 'lower') 
    plt.title ( "%s (GBDT Model)" % state ) 
    plt.xlabel ( "x coordinate [mm]" ) 
    plt.ylabel ( "y coordinate [mm]" ) 
    js = Jscript().hist2d(f"{state}_xy_model", ratio.T, [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]])
    report.add_figure(options=js.width("49%")) ; plt.clf() ; plt.close() 


    ########################## SIMULATED peta ###########################################

    binning = [np.linspace(0,150,150), np.linspace(0.5,6.5,120)] 
    nh = np.zeros ( [149,119], dtype = np.float64 ) 
    dh = np.zeros ( [149,119], dtype = np.float64 ) 
    for db in generated.iterate():
      dh += np.histogram2d ( db['p_%s'%state]*1e-3, db['eta_%s'%state], bins = binning)[0]
    for db in generated.iterate():
      sdb = db.query ( "acceptance == 1")
      nh += np.histogram2d ( sdb['p_%s'%state]*1e-3, sdb['eta_%s'%state], bins = binning)[0]

    ratio = np.where (dh > 0, nh/dh, np.min(nh/(dh+1)))
    plt.imshow (ratio.T, extent = [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]], origin = 'lower', aspect='auto') 
    plt.title ( "%s (Boole/Brunel reconstruction)" % state ) 
    plt.xlabel ( "Momentum [GeV/c]" ) 
    plt.ylabel ( "Pseudorapidity" ) 
    js = Jscript().hist2d(f"{state}_peta_brunel", ratio.T, [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]])
    report.add_figure(options=js.width("49%")) ; plt.clf() ; plt.close() 


    ########################## GENERATED peta ###########################################
    nh = np.zeros ( [149,119], dtype = np.float64 ) 
    dh = np.zeros ( [149,119], dtype = np.float64 ) 

    binning = [np.linspace(0,150,150), np.linspace(0.5,6.5,120)] 
    for db in generated.iterate():
      dh += np.histogram2d ( db['p_%s'%state]*1e-3, db['eta_%s'%state], bins = binning)[0]
    for db in generated.iterate():
      w = classifier.predict_proba( db[accBdt.discrVars].values )[:,1]
      nh += np.histogram2d ( db['p_%s'%state]*1e-3, db['eta_%s'%state], bins = binning, weights=w)[0]

    ratio = np.where (dh > 0, nh/dh, np.min(nh/(dh+1)))
    plt.imshow (ratio.T, extent = [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]], origin = 'lower', aspect='auto') 
    plt.title ( "%s (GBDT Model)" % state ) 
    plt.xlabel ( "Momentum [GeV/c]" ) 
    plt.ylabel ( "Pseudorapidity" ) 
    js = Jscript().hist2d(f"{state}_peta_model", ratio.T, [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]])
    report.add_figure(options=js.width("49%")) ; plt.clf() ; plt.close() 

  report.add_markdown ("### One-dimensional distributions") 
  wd = {
        "Generated": None,
        "Selected":  "acceptance == 1", 
        "Weighted":  lambda db: classifier.predict_proba(db[accBdt.discrVars].values)[:,1] 
      }
  pz_bins = [1e2, 1e3, 2e3, 5e3, 10e3, 50e3]
  eta_bins = [1.8, 2.7, 3.5, 4.2, 5.5] 

  
  ##################  P bins   ###################################################
  report.add_markdown ( "## Momentum bins") 
  selections = [] 
  for pBin, (pz_min, pz_max) in enumerate(zip ( pz_bins[:-1], pz_bins[1:] )): 
    selections.append ((" & ".join ([ 
        "(pz > %f)"  % pz_min,
        "(pz < %f)"  % pz_max, 
        ]),
        "$p_z$ in (%.1f, %.1f) GeV/c"%(pz_min*1e-3, pz_max*1e-3)
        ))

  histmaker = make_histogram (
      cfg.acceptanceBDT.validationHistogramsPBins, 
      cfg.datasets['BrunelGenerated'], 
      weight_dict=wd, 
      errorbars = ["Weighted"],
      selections=selections,
      max_chunks = 10, 
      make_js = True 
      ) 
      
  for axis, js in histmaker: 
    report.add_figure(options=str(js.width('24%'))) ; plt.clf() ; plt.close() 

  ##################  Eta bins   ################################################
  report.add_markdown ( "## Pseudorapidity bins") 
  selections = [] 
  for etaBin, (eta_min, eta_max) in enumerate(zip ( eta_bins[:-1], eta_bins[1:] )): 
    selections.append ((" & ".join ([ 
        "(eta_EndVelo > %f)" % eta_min, 
        "(eta_EndVelo < %f)" % eta_max, 
        ]),
        "$\eta$ in (%.1f, %.1f)"%(eta_min, eta_max)
        ))

  histmaker = make_histogram (
      cfg.acceptanceBDT.validationHistogramsEtaBins, 
      cfg.datasets['BrunelGenerated'], 
      weight_dict=wd, 
      errorbars = ["Weighted"],
      selections=selections, 
      max_chunks = 10, 
      make_js = True,
      ) 
      
  for axis, js in histmaker: 
    report.add_figure(options=str(js.width('24%'))) ; plt.clf() ; plt.close() 


  ##################  PEta bins   ################################################
  report.add_markdown ( "## Momentum-Pseudorapidity bins") 
  selections = [] 
  for pBin, (pz_min, pz_max) in enumerate(zip ( pz_bins[:-1], pz_bins[1:] )): 
    for etaBin, (eta_min, eta_max) in enumerate(zip ( eta_bins[:-1], eta_bins[1:] )): 
      selections.append ((" & ".join ([ 
          "(pz > %f)"  % pz_min,
          "(pz < %f)"  % pz_max, 
          "(eta_EndVelo > %f)" % eta_min, 
          "(eta_EndVelo < %f)" % eta_max, 
          ]),
          "$p_z$ in (%.1f, %.1f) GeV/c; $\eta$ in (%.1f, %.1f)"%(pz_min/1e3, pz_max/1e3, eta_min, eta_max)
          ))

  histmaker = make_histogram (
      cfg.acceptanceBDT.validationHistogramsPEtaBins, 
      cfg.datasets['BrunelGenerated'], 
      weight_dict=wd, 
      errorbars = ["Weighted"],
      selections=selections,
      max_chunks = 10, 
      make_js = True,
      ) 
      
  for axis, js in histmaker: 
    report.add_figure(options=str(js.width('24%'))) ; plt.clf() ; plt.close() 




